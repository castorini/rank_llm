import asyncio
import atexit
import inspect
import uuid
from typing import TYPE_CHECKING, Any

try:
    import vllm
except ImportError:
    vllm = None

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any


class VllmHandler:
    """
    Async vLLM inference handler using AsyncLLMEngine with continuous batching.

    Prompts are submitted one at a time via generate_output_async; the engine
    schedules all concurrently in-flight requests optimally on the GPU so no
    caller needs to wait for other prompts to finish.
    """

    def __init__(
        self,
        model: str,
        download_dir: str,
        enforce_eager: bool,
        max_logprobs: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        **kwargs: Any,
    ):
        if vllm is None:
            raise ImportError("vLLM support requires rank-llm[vllm].")
        engine_args = vllm.AsyncEngineArgs(
            model=model,
            download_dir=download_dir,
            enforce_eager=enforce_eager,
            max_logprobs=max_logprobs,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )
        self._engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self._loop = asyncio.new_event_loop()
        self._closed = False
        self._model = model
        self._tokenizer: PreTrainedTokenizerBase | None = None
        atexit.register(self._shutdown)

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True

        previous_loop = None
        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        try:
            asyncio.set_event_loop(self._loop)
            self._engine.shutdown()
        except Exception:
            pass
        try:
            pending = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass
        finally:
            asyncio.set_event_loop(previous_loop)

    def close(self) -> None:
        self._shutdown()

    def run_coroutine(self, coro: Any) -> Any:
        if self._closed:
            raise RuntimeError("VllmHandler is closed.")
        previous_loop = None
        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        try:
            asyncio.set_event_loop(self._loop)
            return self._loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(previous_loop)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            tokenizer_or_coro = self._engine.get_tokenizer()
            if inspect.isawaitable(tokenizer_or_coro):
                self._tokenizer = self.run_coroutine(tokenizer_or_coro)
            else:
                self._tokenizer = tokenizer_or_coro
            if "rank_vicuna" in self._model:
                self._tokenizer.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\n                    {% for message in messages %}{% if not loop.first %}{% endif %}{% if message['role'] == 'system' %}{{ message['content'] + ' ' }}{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"""
        return self._tokenizer

    async def generate_output_async(
        self,
        prompt: str | list[dict[str, str]],
        min_tokens: int,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = -1,
        logprobs: int | None = None,
    ) -> tuple[str, int, int]:
        """
        Submit a single prompt and await its completion.
        Returns (output_text, prompt_token_count, completion_token_count).
        """
        sampling_params = vllm.SamplingParams(
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=logprobs,
        )
        request_id = str(uuid.uuid4())
        output_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        async for request_output in self._engine.generate(
            prompt, sampling_params, request_id
        ):
            if request_output.finished:
                output = request_output.outputs[0]
                output_text = output.text
                prompt_tokens = len(request_output.prompt_token_ids)
                completion_tokens = len(output.token_ids)
        return output_text, prompt_tokens, completion_tokens

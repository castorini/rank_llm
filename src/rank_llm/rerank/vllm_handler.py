import asyncio
import atexit
import uuid
from typing import Any, Dict, List, Optional, Tuple

import vllm
from transformers import PreTrainedTokenizerBase


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
        self._model = model
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        atexit.register(self._shutdown)

    def _shutdown(self) -> None:
        try:
            self._engine.shutdown()
        except Exception:
            pass

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = asyncio.run(self._engine.get_tokenizer())
            if "rank_vicuna" in self._model:
                setattr(
                    self._tokenizer,
                    "chat_template",
                    """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
                    {% for message in messages %}{% if not loop.first %}{% endif %}{% if message['role'] == 'system' %}{{ message['content'] + ' ' }}{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}""",
                )
        return self._tokenizer

    async def generate_output_async(
        self,
        prompt: str | List[Dict[str, str]],
        min_tokens: int,
        max_tokens: int,
        temperature: float,
        logprobs: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        """
        Submit a single prompt and await its completion.
        Returns (output_text, prompt_token_count, completion_token_count).
        """
        sampling_params = vllm.SamplingParams(
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
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

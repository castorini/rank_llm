from typing import Any, Dict, List, Optional

try:
    import vllm
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    vllm = None
    RequestOutput = None

try:
    from transformers import PreTrainedTokenizerBase
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedTokenizerBase = None


class VllmHandler:
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
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install it with: pip install rank_llm[vllm]"
            )
        
        self._vllm = vllm.LLM(
            model=model,
            download_dir=download_dir,
            enforce_eager=enforce_eager,
            max_logprobs=max_logprobs,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._tokenizer = self._vllm.get_tokenizer()

        if "rank_vicuna" in model:
            setattr(
                self._tokenizer,
                "chat_template",
                """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
                    {% for message in messages %}{% if not loop.first %}{% endif %}{% if message['role'] == 'system' %}{{ message['content'] + ' ' }}{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}""",
            )

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. Please install it with: pip install rank_llm[transformers]"
            )
        return self._tokenizer

    def generate_output(
        self,
        prompts: List[str | List[Dict[str, str]]],
        min_tokens: int,
        max_tokens: int,
        temperature: float,
        logprobs: Optional[int] = None,
        **kwargs: Any,
    ) -> List[RequestOutput]:
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install it with: pip install rank_llm[vllm]"
            )
        
        # TODO: Implement rest of vllm arguments (from kwargs) in the future if necessary
        sampling_params = vllm.SamplingParams(
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            **kwargs,
        )

        return self._vllm.generate(prompts, sampling_params)

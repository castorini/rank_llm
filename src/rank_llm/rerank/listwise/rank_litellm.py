import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from rank_llm._optional import missing_extra_error
from rank_llm.data import Request, Result
from rank_llm.rerank.rankllm import PromptMode

from .listwise_rankllm import ListwiseRankLLM

try:
    import litellm
except ImportError:
    litellm = None


class SafeLiteLLM(ListwiseRankLLM):
    """Listwise reranker using LiteLLM for 100+ LLM providers.

    Accepts any LiteLLM model string (e.g. ``openai/gpt-4o``,
    ``anthropic/claude-sonnet-4-6``, ``groq/llama-3.3-70b-versatile``)
    and uses ``litellm.completion()`` for the chat completions call.

    Parameters are identical to ``SafeOpenai`` except:
    - ``model`` accepts any LiteLLM model string
    - ``api_key`` is a single key (no key cycling)
    - ``api_base`` is optional (only needed for custom endpoints)
    - ``sampling_kwargs`` optional dict of sampling overrides
    """

    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode | None = None,
        prompt_template_path: str | None = None,
        num_few_shot_examples: int = 0,
        few_shot_file: str | None = None,
        window_size: int = 20,
        stride: int = 10,
        batch_size: int = 32,
        api_key: str | None = None,
        api_base: str | None = None,
        max_passage_words: int = 300,
        sampling_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if litellm is None:
            raise missing_extra_error(
                "litellm",
                "The LiteLLM reranker requires the litellm package.",
            )

        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            max_passage_words=max_passage_words,
        )

        self._output_token_estimate = None
        self._api_key = api_key
        self._api_base = api_base
        self._sampling_kwargs: dict[str, Any] | None = (
            dict(sampling_kwargs) if sampling_kwargs else None
        )

    def _call_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {"model": self._model, "drop_params": True}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._sampling_kwargs:
            kwargs.update(self._sampling_kwargs)
        return kwargs

    def rerank_batch(
        self,
        requests: list[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> list[Result]:
        top_k_retrieve: int = kwargs.get("top_k_retrieve", rank_end)
        rank_end = min(top_k_retrieve, rank_end)
        populate_invocations_history: bool = kwargs.get(
            "populate_invocations_history", False
        )
        if not requests:
            return []

        worker_cap = (
            self._batch_size
            if self._batch_size and self._batch_size > 0
            else len(requests)
        )
        max_workers = min(len(requests), max(worker_cap, 1))
        if max_workers <= 1:
            results: list[Result] = []
            for request in tqdm(requests):
                result = self.sliding_windows(
                    request,
                    rank_start=max(rank_start, 0),
                    rank_end=min(rank_end, len(request.candidates)),
                    top_k_retrieve=top_k_retrieve,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_invocations_history=populate_invocations_history,
                )
                results.append(result)
            return results

        results: dict[int, Result] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.sliding_windows,
                    request,
                    rank_start=max(rank_start, 0),
                    rank_end=min(rank_end, len(request.candidates)),
                    top_k_retrieve=top_k_retrieve,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_invocations_history=populate_invocations_history,
                ): index
                for index, request in enumerate(requests)
            }
            progress = tqdm(total=len(requests))
            try:
                for future in as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                    progress.update(1)
            finally:
                progress.close()

        return [results[index] for index in range(len(requests))]

    async def rerank_batch_async(
        self,
        requests: list[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> list[Result]:
        return await asyncio.to_thread(
            self.rerank_batch,
            requests,
            rank_start,
            rank_end,
            shuffle_candidates,
            logging,
            **kwargs,
        )

    def _call_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> Any:
        call_kwargs = {**self._call_kwargs(), **kwargs}
        try:
            return litellm.completion(messages=messages, timeout=300, **call_kwargs)
        except Exception as e:
            print("Error in completion call")
            print(str(e))
            if "maximum context length" in str(e).lower():
                print("reduce_length")
                return "ERROR::reduce_length"
            if "response was filtered" in str(e).lower():
                print("The response was filtered")
                return "ERROR::The response was filtered"
            raise

    def run_llm(
        self,
        prompt: str | list[dict[str, str]],
        current_window_size: int | None = None,
    ) -> tuple[str, str | None, dict[str, Any]]:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        completion = self._call_completion(messages)

        if isinstance(completion, str):
            return completion, None, {}

        text = completion.choices[0].message.content

        usage = completion.usage if completion.usage is not None else {}

        return text, None, usage

    def num_output_tokens(self, current_window_size: int | None = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate

        output_str = " > ".join([f"[{i + 1}]" for i in range(current_window_size)])
        _output_token_estimate = litellm.token_counter(
            model=self._model, text=output_str
        )
        if (
            self._output_token_estimate is None
            and self._window_size == current_window_size
        ):
            self._output_token_estimate = _output_token_estimate
        return _output_token_estimate

    def create_prompt_batched(self):
        pass

    def run_llm_batched(self):
        pass

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> tuple[list[dict[str, str]], int]:
        max_length = self._max_passage_words

        while True:
            prompt = self._inference_handler.generate_prompt(
                result=result,
                rank_start=rank_start,
                rank_end=rank_end,
                max_length=max_length,
                num_fewshot_examples=self._num_few_shot_examples,
                fewshot_examples=self._examples,
            )
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // ((rank_end - rank_start) * 4),
                )

        return prompt, num_tokens

    def get_num_tokens(self, prompt: str | list[dict[str, str]]) -> int:
        if isinstance(prompt, list):
            return litellm.token_counter(model=self._model, messages=prompt)
        return litellm.token_counter(model=self._model, text=prompt)

    def cost_per_1k_token(self, input_token: bool) -> float:
        try:
            info = litellm.get_model_cost_map(url="").get(self._model, {})
            if input_token:
                return info.get("input_cost_per_token", 0) * 1000
            return info.get("output_cost_per_token", 0) * 1000
        except Exception:
            return 0.0

    def get_name(self) -> str:
        return self._model

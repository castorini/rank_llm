from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class VllmHandlerWithOpenAISDK:
    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        batch_size: int = 16,
    ):
        self._client = OpenAI(api_key="EMPTY", base_url=base_url)

        # if model isn't provided, use the SYNC client to list models
        if model is None:
            models = self._client.models.list()
            if not models.data:
                raise RuntimeError("No models available from vLLM /v1/models.")
            model = models.data[0].id

        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._executor = ThreadPoolExecutor(max_workers=batch_size)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def _one_inference(
        self, messages: list[dict[str, str]], **kwargs
    ) -> Tuple[str, str, Dict[str, Any]]:
        assert isinstance(messages, list)
        assert isinstance(messages[0], dict)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **kwargs,
            )
            text = response.choices[0].message.content
            reasoning = response.choices[0].message.reasoning
            usage = response.usage.model_dump(mode="json")
            return text, reasoning, usage
        except Exception as e:
            print(f"Error during inference: {e}")
            return str(e), "", {}

    def chat_completions(
        self, prompts: list[list[dict[str, str]]], **kwargs
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        return list(
            self._executor.map(lambda p: self._one_inference(p, **kwargs), prompts)
        )

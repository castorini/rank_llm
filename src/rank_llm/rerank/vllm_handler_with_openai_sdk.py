from typing import Any, Dict, Tuple

from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class VllmHandlerWithOpenAISDK:
    """
    Async OpenAI-compatible inference handler.

    Uses AsyncOpenAI for inference so all concurrently submitted coroutines
    are in-flight simultaneously. The sync OpenAI client is kept only for the
    one-time model discovery call at init time.
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
    ):
        sync_client = OpenAI(api_key="EMPTY", base_url=base_url)
        self._async_client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)

        if model is None:
            models = sync_client.models.list()
            if not models.data:
                raise RuntimeError("No models available from vLLM /v1/models.")
            model = models.data[0].id

        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(model)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    async def chat_completion_async(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Submit a single chat request and await its completion."""
        try:
            response = await self._async_client.chat.completions.create(
                model=self._model,
                messages=messages,
                **kwargs,
            )
            text = response.choices[0].message.content
            reasoning = response.choices[0].message.reasoning
            usage = response.usage.model_dump(mode="json")
            return text, reasoning, usage
        except Exception as e:
            print(f"Error during async inference: {e}")
            return str(e), "", {}

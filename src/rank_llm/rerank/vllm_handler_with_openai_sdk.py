from typing import TYPE_CHECKING, Any

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    AsyncOpenAI = None
    OpenAI = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any


class VllmHandlerWithOpenAISDK:
    """
    Async OpenAI-compatible inference handler.

    Uses AsyncOpenAI for inference so all concurrently submitted coroutines
    are in-flight simultaneously. The sync OpenAI client is kept only for the
    one-time model discovery call at init time.

    Unlike :class:`~rank_llm.rerank.vllm_handler.VllmHandler`, the tokenizer is
    loaded synchronously in ``__init__`` (no ``asyncio.run``). It is therefore
    safe to construct this handler while an asyncio event loop is already
    running (e.g. building ``RankListwiseOSLLM`` from inside an async task).
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
    ):
        if any(dep is None for dep in (OpenAI, AsyncOpenAI, AutoTokenizer)):
            raise ImportError("OpenAI-compatible vLLM support requires rank-llm[vllm].")
        sync_client = OpenAI(api_key="EMPTY", base_url=base_url)
        self._async_client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)

        if model is None:
            models = sync_client.models.list()
            if not models.data:
                raise RuntimeError("No models available from vLLM /v1/models.")
            model = models.data[0].id

        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True
        )

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer loaded in ``__init__`` (always sync, loop-safe)."""
        return self._tokenizer

    async def chat_completion_async(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> tuple[str, str, dict[str, Any]]:
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

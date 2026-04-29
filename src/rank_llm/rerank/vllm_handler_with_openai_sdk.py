import math
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

# First-completion-token logprob bucketing for binary pointwise scoring.
# Match stripped API token strings (vLLM may return different casings).
POINTWISE_YES_LOGPROB_TOKEN_STRINGS: tuple[str, ...] = ("yes", "Yes", "YES")
POINTWISE_NO_LOGPROB_TOKEN_STRINGS: tuple[str, ...] = ("no", "No", "NO")


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
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

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

    @staticmethod
    def score_from_top_logprobs(
        top_logprobs: list[dict[str, Any]] | None,
        fallback_lp: float = -20.0,
    ) -> tuple[float, float, float]:
        """Normalize yes/no mass from the first completion token's top logprobs."""
        total_yes_prob = 0.0
        total_no_prob = 0.0

        if top_logprobs:
            for e in top_logprobs:
                tok = (e.get("token") or "").strip()
                lp = e.get("logprob")
                if lp is None:
                    continue
                prob = math.exp(float(lp))
                if tok in POINTWISE_YES_LOGPROB_TOKEN_STRINGS:
                    total_yes_prob += prob
                elif tok in POINTWISE_NO_LOGPROB_TOKEN_STRINGS:
                    total_no_prob += prob

        if total_yes_prob == 0.0 and total_no_prob == 0.0:
            return 0.0, fallback_lp, fallback_lp

        score = total_yes_prob / (total_yes_prob + total_no_prob)
        yes_lp = math.log(total_yes_prob) if total_yes_prob > 0 else fallback_lp
        no_lp = math.log(total_no_prob) if total_no_prob > 0 else fallback_lp
        return score, yes_lp, no_lp

    async def chat_completion_score_async(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = True,
        top_logprobs: int = 20,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[str, int, float, dict[str, Any]]:
        """Single-token chat completion with logprobs; returns (text, out_tokens, score, usage)."""
        body = dict(kwargs)
        body.setdefault("max_tokens", max_tokens)
        body.setdefault("temperature", temperature)
        body.setdefault("logprobs", logprobs)
        body.setdefault("top_logprobs", top_logprobs)
        if extra_body is not None:
            body["extra_body"] = {**(body.get("extra_body") or {}), **extra_body}

        try:
            response = await self._async_client.chat.completions.create(
                model=self._model,
                messages=messages,
                **body,
            )
        except Exception as e:
            print(f"Error during async score inference: {e}")
            return str(e), 0, -1.0, {}

        usage = response.usage.model_dump(mode="json") if response.usage else {}
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        out_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )

        top_lp: list[dict[str, Any]] = []
        if choice.logprobs and choice.logprobs.content:
            token_lps = choice.logprobs.content[0].top_logprobs or []
            top_lp = [{"token": lp.token, "logprob": lp.logprob} for lp in token_lps]

        score, _, _ = self.score_from_top_logprobs(top_lp)
        return text, out_tokens, float(score), usage

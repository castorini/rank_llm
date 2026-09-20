import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
# ``token`` strings from the API are matched after ``str.strip()`` so values
# like ``"yes "`` or ``" no"`` still map to the same class as ``"yes"`` / ``"no"``.
POINTWISE_YES_LOGPROB_TOKEN_STRINGS: tuple[str, ...] = ("yes", "Yes", "YES")
POINTWISE_NO_LOGPROB_TOKEN_STRINGS: tuple[str, ...] = ("no", "No", "NO")

POINTWISE_YES_LOGPROB_STRIPPED: frozenset[str] = frozenset(
    {s.strip() for s in POINTWISE_YES_LOGPROB_TOKEN_STRINGS}
)
POINTWISE_NO_LOGPROB_STRIPPED: frozenset[str] = frozenset(
    {s.strip() for s in POINTWISE_NO_LOGPROB_TOKEN_STRINGS}
)

# Extra spellings tried for ``allowed_token_ids`` only (each must be one token).
# Logprob bucketing uses stripped strings, so ``"yes "`` in API responses still
# scores as yes; if the tokenizer maps ``"yes "`` to a *different* single id
# than ``"yes"``, include that spelling here so both can be sampled.
POINTWISE_YES_ALLOWED_TOKEN_SPELLINGS: tuple[str, ...] = (
    *POINTWISE_YES_LOGPROB_TOKEN_STRINGS,
    "yes ",
    "Yes ",
    "YES ",
    " yes",
    " Yes",
    " YES",
)
POINTWISE_NO_ALLOWED_TOKEN_SPELLINGS: tuple[str, ...] = (
    *POINTWISE_NO_LOGPROB_TOKEN_STRINGS,
    "no ",
    "No ",
    "NO ",
    " no",
    " No",
    " NO",
)

logger = logging.getLogger(__name__)

# With vLLM's per-request ``return_tokens_as_token_ids``, logprob entries name
# their token as ``token_id:<id>`` instead of its text.
TOKEN_ID_PREFIX = "token_id:"


def single_token_ids(
    tokenizer: Any, spellings: Sequence[str]
) -> tuple[list[int], list[str]]:
    """Token ids of the spellings that encode to exactly one token, and the skipped ones.

    A pointwise judge answers with one token, so a spelling that the tokenizer
    splits into several pieces cannot be its answer and is skipped.
    """
    ids: list[int] = []
    skipped: list[str] = []
    for spelling in spellings:
        encoded = tokenizer.encode(spelling, add_special_tokens=False)
        if len(encoded) != 1:
            skipped.append(spelling)
        elif encoded[0] not in ids:
            ids.append(encoded[0])
    return ids, skipped


@dataclass(frozen=True)
class PointwiseAnswerTokens:
    """The tokens a pointwise judge may answer with, split into positive and negative.

    Configured as spellings, e.g. ``("yes",)`` / ``("no",)`` for the published
    Qwen3-Reranker recipe or ``("true",)`` / ``("false",)`` for a model prompted
    to answer true or false, and resolved once to token ids with the served
    model's tokenizer. ``allowed_token_ids`` is the union of both sides, and a
    score is the probability mass on positive ids over the mass on both, read
    from the first completion token's top logprobs. Matching is by token id, so
    no case or whitespace rule is applied to the server's token strings: the
    spellings alone decide what counts.
    """

    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...]
    # Logprob token string -> id: "token_id:<id>" (what vLLM returns when the
    # request sets return_tokens_as_token_ids), plus each id's decoded text as
    # a fallback for servers that ignore that field.
    lookup: Mapping[str, int] = field(repr=False, compare=False)

    @classmethod
    def from_spellings(
        cls,
        tokenizer: Any,
        positive: Sequence[str],
        negative: Sequence[str],
        *,
        warn_skipped: bool = True,
    ) -> "PointwiseAnswerTokens":
        sides = {}
        for name, spellings in (("positive", positive), ("negative", negative)):
            ids, skipped = single_token_ids(tokenizer, spellings)
            if skipped:
                log = logger.warning if warn_skipped else logger.debug
                log(
                    "Skipping %s answer spellings that are not a single token: %s",
                    name,
                    skipped,
                )
            if not ids:
                raise ValueError(
                    f"No {name} answer spelling is a single token for this "
                    f"tokenizer: {list(spellings)}"
                )
            sides[name] = ids
        overlap = set(sides["positive"]) & set(sides["negative"])
        if overlap:
            raise ValueError(
                f"Token ids {sorted(overlap)} are both positive and negative answers"
            )
        answer_ids = sides["positive"] + sides["negative"]
        lookup: dict[str, int] = {f"{TOKEN_ID_PREFIX}{i}": i for i in answer_ids}
        decode = getattr(tokenizer, "decode", None)
        if decode is not None:
            texts: dict[str, set[int]] = {}
            for i in answer_ids:
                texts.setdefault(decode([i]), set()).add(i)
            for text, ids in texts.items():
                if len(ids) == 1:  # an ambiguous text cannot name an id
                    lookup.setdefault(text, next(iter(ids)))
        return cls(tuple(sides["positive"]), tuple(sides["negative"]), lookup)

    @property
    def allowed_token_ids(self) -> list[int]:
        return [*self.positive_ids, *self.negative_ids]

    def score(
        self,
        top_logprobs: list[dict[str, Any]] | None,
        fallback_lp: float = -20.0,
    ) -> tuple[float, float, float]:
        """(score, positive logprob, negative logprob) from the first token's top logprobs."""
        positive = set(self.positive_ids)
        negative = set(self.negative_ids)
        positive_prob = negative_prob = 0.0
        for entry in top_logprobs or []:
            token_id = self.lookup.get(entry.get("token") or "")
            lp = entry.get("logprob")
            if token_id is None or lp is None:
                continue
            if token_id in positive:
                positive_prob += math.exp(float(lp))
            elif token_id in negative:
                negative_prob += math.exp(float(lp))
        if positive_prob == 0.0 and negative_prob == 0.0:
            return 0.0, fallback_lp, fallback_lp
        score = positive_prob / (positive_prob + negative_prob)
        positive_lp = math.log(positive_prob) if positive_prob > 0 else fallback_lp
        negative_lp = math.log(negative_prob) if negative_prob > 0 else fallback_lp
        return score, positive_lp, negative_lp


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
        self._base_url = base_url
        self._async_client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        self._async_client_loop: asyncio.AbstractEventLoop | None = None
        # Clients retired by a loop change are kept, not dropped: garbage
        # collection would schedule their aclose() on the closed loop, which
        # only produces "Task exception was never retrieved" noise.
        self._retired_async_clients: list[Any] = []

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

    def _async_client_for_current_loop(self) -> Any:
        """The async client bound to the running event loop.

        Callers drive each batch with ``asyncio.run``, which closes its loop on
        return. The client's pooled HTTP connections belong to the loop they
        were opened on, so reusing the client from the next loop fails with
        "Event loop is closed" whenever a connection is still alive in the
        pool, i.e. whenever batches arrive faster than the keep-alive expiry.
        """
        loop = asyncio.get_running_loop()
        if self._async_client_loop is None:
            self._async_client_loop = loop
        elif self._async_client_loop is not loop:
            self._retired_async_clients.append(self._async_client)
            self._async_client = AsyncOpenAI(api_key="EMPTY", base_url=self._base_url)
            self._async_client_loop = loop
        return self._async_client

    async def chat_completion_async(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> tuple[str, str, dict[str, Any]]:
        """Submit a single chat request and await its completion.

        ``extra_body`` (and any other vLLM-specific keys) flow through
        ``**kwargs``; they are only forwarded when the caller actually sets them
        so the underlying OpenAI client receives the same call shape as before
        for callers that pass no extras.
        """
        try:
            response = await self._async_client_for_current_loop().chat.completions.create(
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
                if tok in POINTWISE_YES_LOGPROB_STRIPPED:
                    total_yes_prob += prob
                elif tok in POINTWISE_NO_LOGPROB_STRIPPED:
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
        answer_tokens: PointwiseAnswerTokens | None = None,
        **kwargs: Any,
    ) -> tuple[str, int, float, dict[str, Any]]:
        """Single-token chat completion with logprobs; returns (text, out_tokens, score, usage).

        With ``answer_tokens`` the logprobs are requested as token ids and scored
        by them; without, the yes/no spellings above are matched as stripped text.
        """
        body = dict(kwargs)
        body.setdefault("max_tokens", max_tokens)
        body.setdefault("temperature", temperature)
        body.setdefault("logprobs", logprobs)
        body.setdefault("top_logprobs", top_logprobs)
        if extra_body is not None:
            body["extra_body"] = {**(body.get("extra_body") or {}), **extra_body}
        if answer_tokens is not None:
            body["extra_body"] = {
                **(body.get("extra_body") or {}),
                "return_tokens_as_token_ids": True,
            }

        try:
            response = await self._async_client_for_current_loop().chat.completions.create(
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

        if answer_tokens is not None:
            score, _, _ = answer_tokens.score(top_lp)
        else:
            score, _, _ = self.score_from_top_logprobs(top_lp)
        return text, out_tokens, float(score), usage

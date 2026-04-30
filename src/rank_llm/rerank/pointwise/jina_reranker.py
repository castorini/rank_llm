import copy
import logging
from functools import cmp_to_key
from importlib.resources import files
from typing import Any

from ftfy import fix_text
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from rank_llm.data import InferenceInvocation, Request, Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM

logger = logging.getLogger(__name__)

TEMPLATES = files("rank_llm.rerank.prompt_templates")

MAX_JINA_DOCS_PER_CALL = 64
TOKENS_PER_WORD_RATIO = 1.34
WORDS_PER_TOKEN_RATIO = 0.75
QUERY_OVERHEAD_TOKENS = 128
BASE_OUTPUT_RESERVE_TOKENS = 64
PER_DOC_INPUT_OVERHEAD_TOKENS = 32
PER_DOC_OUTPUT_OVERHEAD_TOKENS = 8
MIN_WORDS_PER_DOC = 16
MIN_USER_PASSAGE_WORDS = 1
MIN_DOCS_PER_CHUNK = 1
MIN_PASSAGE_LENGTH_DECREMENT = 1


def _extract_doc_text(doc: dict[str, Any], max_words: int | None = None) -> str:
    """Extract and clean document text, matching the project-wide field precedence."""
    if "text" in doc:
        content = doc["text"]
    elif "segment" in doc:
        content = doc["segment"]
    elif "contents" in doc:
        content = doc["contents"]
    elif "content" in doc:
        content = doc["content"]
    elif "body" in doc:
        content = doc["body"]
    elif "passage" in doc:
        content = doc["passage"]
    else:
        content = str(doc)

    if "title" in doc and doc["title"]:
        content = "Title: " + doc["title"] + " Content: " + content

    content = content.strip()
    content = fix_text(content)

    if max_words is not None:
        content = " ".join(content.split()[:max_words])

    return content


def _truncate_to_max_words(text: str, max_words: int | None) -> str:
    if max_words is None:
        return text
    return " ".join(text.split()[:max_words])


class JinaReranker(PointwiseRankLLM):
    """Local reranker using Jina Reranker v3 (``jinaai/jina-reranker-v3``).

    Jina v3 is a 0.6B-param encoder model that scores up to 64 documents per
    call using causal self-attention between query and documents.  It produces
    per-document relevance scores (like pointwise) but benefits from
    cross-document attention within a single forward pass (like listwise).

    When a query has more candidates than fit in one call the candidates are
    chunked.  Jina scores are absolute (not relative to the batch) so raw
    scores are used directly without any cross-chunk normalisation.

    Args:
        model: HuggingFace model ID, default ``"jinaai/jina-reranker-v3"``.
        context_size: Model context window in tokens (131 072 for Jina v3).
        device: ``"cuda"`` or ``"cpu"``.
        window_size: Max documents per ``model.rerank()`` call (capped at 64).
        dtype: Torch dtype string passed to ``from_pretrained``.
        max_passage_words: Preferred per-passage truncation length in words.
            This is treated as an upper bound: if ``window_size`` passages at
            this length do not fit in ``context_size`` (including output
            reservation), the passage length is reduced further until the chunk
            fits. When ``None``, the initial limit is derived from
            ``context_size / window_size`` and then tightened until it fits.
    """

    def __init__(
        self,
        model: str = "jinaai/jina-reranker-v3",
        context_size: int = 131_072,
        prompt_template_path: str = (TEMPLATES / "jina_template.yaml"),
        device: str = "cuda",
        window_size: int = MAX_JINA_DOCS_PER_CALL,
        batch_size: int = 1,
        dtype: str = "auto",
        max_passage_words: int | None = None,
    ):
        window_size = min(window_size, MAX_JINA_DOCS_PER_CALL)

        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=None,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=0,
            few_shot_file=None,
            device=device,
            batch_size=batch_size,
        )

        self._llm = AutoModel.from_pretrained(
            model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        tokenizer = getattr(self._llm, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=True,
            )
        self._tokenizer = tokenizer
        self._llm.eval()
        self._llm.to(self._device)

        self._dtype = dtype
        self._window_size = window_size
        self._max_passage_words = max_passage_words

    def _estimate_per_doc_total_tokens(self, avg_doc_words: int) -> int:
        per_doc_word_tokens = int(avg_doc_words * TOKENS_PER_WORD_RATIO)
        per_doc_overhead_tokens = (
            PER_DOC_INPUT_OVERHEAD_TOKENS + PER_DOC_OUTPUT_OVERHEAD_TOKENS
        )
        return max(per_doc_word_tokens + per_doc_overhead_tokens, 1)

    def _available_context_tokens(self) -> int:
        available_tokens = (
            self._context_size - QUERY_OVERHEAD_TOKENS - BASE_OUTPUT_RESERVE_TOKENS
        )
        return max(available_tokens, 1)

    def _compute_effective_max_words(self, num_docs: int) -> int:
        """Derive a context-aware per-document word limit."""
        safe_num_docs = max(num_docs, 1)
        available_tokens = self._available_context_tokens()
        per_doc_overhead_tokens = (
            PER_DOC_INPUT_OVERHEAD_TOKENS + PER_DOC_OUTPUT_OVERHEAD_TOKENS
        )
        per_doc_content_tokens = max(
            (available_tokens // safe_num_docs) - per_doc_overhead_tokens, 1
        )
        words_per_doc = int(per_doc_content_tokens * WORDS_PER_TOKEN_RATIO)
        return max(words_per_doc, MIN_WORDS_PER_DOC)

    def _compute_docs_per_chunk(self, avg_doc_words: int) -> int:
        """Return how many docs fit at this passage length, capped by window size."""
        per_doc_tokens = self._estimate_per_doc_total_tokens(max(avg_doc_words, 1))
        available_tokens = self._available_context_tokens()
        fit = max(available_tokens // per_doc_tokens, MIN_DOCS_PER_CHUNK)
        return min(fit, self._window_size)

    def _fit_max_passage_words(self, num_docs: int) -> int:
        """Find a passage-length cap that fits context for this chunk size.

        Starts from user-provided ``max_passage_words`` when available; if that
        does not fit ``context_size`` for ``num_docs`` (including output budget),
        iteratively shrinks until it does.
        """
        safe_num_docs = max(num_docs, 1)
        if self._max_passage_words is not None:
            max_words = max(self._max_passage_words, MIN_USER_PASSAGE_WORDS)
            min_words = MIN_USER_PASSAGE_WORDS
        else:
            max_words = self._compute_effective_max_words(safe_num_docs)
            min_words = MIN_WORDS_PER_DOC

        available_tokens = self._available_context_tokens()
        while (
            safe_num_docs * self._estimate_per_doc_total_tokens(max_words)
            > available_tokens
            and max_words > min_words
        ):
            total_tokens = safe_num_docs * self._estimate_per_doc_total_tokens(
                max_words
            )
            scaled_words = max(
                int(max_words * available_tokens / max(total_tokens, 1)),
                min_words,
            )
            if scaled_words >= max_words:
                scaled_words = max(max_words - MIN_PASSAGE_LENGTH_DECREMENT, min_words)
            max_words = scaled_words

        return max_words

    def _count_tokens(self, text: str) -> int:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_ids = encoded.get("input_ids", [])
        if token_ids and isinstance(token_ids[0], list):
            return sum(len(ids) for ids in token_ids)
        return len(token_ids)

    def _compute_chunk_input_tokens(
        self, query_text: str, chunk_docs: list[str]
    ) -> int:
        input_text = (
            "Query: "
            + query_text
            + "\n"
            + "\n".join(f"Document {idx}: {doc}" for idx, doc in enumerate(chunk_docs))
        )
        return self._count_tokens(input_text)

    def rerank_batch(
        self,
        requests: list[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> list[Result]:
        populate_invocations_history: bool = kwargs.get(
            "populate_invocations_history", False
        )

        rerank_results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                invocations_history=[],
            )
            for request in requests
        ]

        with tqdm(
            total=len(rerank_results), desc="Jina reranking queries"
        ) as progress_bar:
            # Run the reranking in batches of size self._batch_size sequentially since the Jina reranker does not support batching.
            for batch_start in range(0, len(rerank_results), self._batch_size):
                batch_end = min(batch_start + self._batch_size, len(rerank_results))
                batch_results = rerank_results[batch_start:batch_end]

                for idx, result in enumerate(batch_results):
                    batch_results[idx] = self._rerank_single_result(
                        result, populate_invocations_history
                    )
                    progress_bar.update(1)

        return rerank_results

    def _rerank_single_result(
        self,
        result: Result,
        populate_invocations_history: bool,
    ) -> Result:
        query_text = result.query.text
        candidates = result.candidates
        num_candidates = len(candidates)
        doc_texts_raw = [_extract_doc_text(c.doc, max_words=None) for c in candidates]

        all_scores: list[float] = [0.0] * num_candidates

        chunk_start = 0
        while chunk_start < num_candidates:
            remaining = num_candidates - chunk_start
            target_docs = min(remaining, self._window_size)
            effective_max_words = self._fit_max_passage_words(self._window_size)
            chunk_size = target_docs
            chunk_end = min(chunk_start + chunk_size, num_candidates)
            chunk_docs = [
                _truncate_to_max_words(doc_texts_raw[i], effective_max_words)
                for i in range(chunk_start, chunk_end)
            ]

            jina_results = self._llm.rerank(query_text, chunk_docs)

            for item in jina_results:
                all_scores[chunk_start + item["index"]] = float(item["relevance_score"])

            if populate_invocations_history:
                chunk_scores = [0.0] * len(chunk_docs)
                for item in jina_results:
                    chunk_scores[item["index"]] = float(item["relevance_score"])
                prompt_repr = (
                    f"query={query_text!r}, "
                    f"docs[{chunk_start}:{chunk_end}] "
                    f"({len(chunk_docs)} docs)"
                )
                response_repr = ", ".join(
                    f"[{chunk_start + i}]={s:.4f}" for i, s in enumerate(chunk_scores)
                )
                result.invocations_history.append(
                    InferenceInvocation(
                        prompt=prompt_repr,
                        response=response_repr,
                        input_token_count=self._compute_chunk_input_tokens(
                            query_text=query_text, chunk_docs=chunk_docs
                        ),
                        output_token_count=0,
                    )
                )
            chunk_start = chunk_end

        for idx, score in enumerate(all_scores):
            result.candidates[idx].score = score

        result.candidates.sort(key=cmp_to_key(self.candidate_comparator), reverse=True)
        return result

    # -- Abstract method implementations (not used in primary rerank path) ----

    def run_llm_batched(
        self,
        prompts: list[str],
    ) -> tuple[list[str], list[int], list[float]]:
        raise NotImplementedError(
            "JinaReranker uses model.rerank() directly; see rerank_batch() instead."
        )

    def run_llm(self, prompt: str) -> tuple[str, int, float]:
        raise NotImplementedError(
            "JinaReranker uses model.rerank() directly; see rerank_batch() instead."
        )

    def create_prompt(self, result: Result, index: int) -> tuple[str, int]:
        raise NotImplementedError(
            "JinaReranker bypasses prompt creation and uses model.rerank() directly."
        )

    def get_num_tokens(self, prompt: str) -> int:
        return self._count_tokens(prompt)

    def num_output_tokens(self) -> int:
        return 0

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

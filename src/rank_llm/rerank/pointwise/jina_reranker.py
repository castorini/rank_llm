import copy
import logging
import math
from functools import cmp_to_key
from importlib.resources import files
from typing import Any

from ftfy import fix_text
from tqdm import tqdm

from transformers import AutoModel

from rank_llm.data import InferenceInvocation, Request, Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM

logger = logging.getLogger(__name__)

TEMPLATES = files("rank_llm.rerank.prompt_templates")

MAX_JINA_DOCS_PER_CALL = 64


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


class JinaReranker(PointwiseRankLLM):
    """Local reranker using Jina Reranker v3 (``jinaai/jina-reranker-v3``).

    Jina v3 is a 0.6B-param encoder model that scores up to 64 documents per
    call using causal self-attention between query and documents.  It produces
    per-document relevance scores (like pointwise) but benefits from
    cross-document attention within a single forward pass (like listwise).

    When a query has more candidates than fit in one call, candidates are
    chunked and scores are min-max normalised within each chunk so that
    cross-chunk scores are comparable.

    Args:
        model: HuggingFace model ID, default ``"jinaai/jina-reranker-v3"``.
        context_size: Model context window in tokens (131 072 for Jina v3).
        device: ``"cuda"`` or ``"cpu"``.
        batch_size: Max documents per ``model.rerank()`` call (capped at 64).
        dtype: Torch dtype string passed to ``from_pretrained``.
        max_passage_words: If set, truncate each passage to this many
            whitespace-delimited words *before* sending to the model.  When
            ``None`` the effective limit is derived from
            ``context_size / num_docs`` (in tokens, approximated as words/0.75).
    """

    def __init__(
        self,
        model: str = "jinaai/jina-reranker-v3",
        context_size: int = 131_072,
        prompt_template_path: str = (TEMPLATES / "jina_template.yaml"),
        device: str = "cuda",
        batch_size: int = MAX_JINA_DOCS_PER_CALL,
        dtype: str = "auto",
        max_passage_words: int | None = None,
    ):
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=None,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=0,
            few_shot_file=None,
            device=device,
            batch_size=min(batch_size, MAX_JINA_DOCS_PER_CALL),
        )

        self._llm = AutoModel.from_pretrained(
            model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._llm.eval()
        self._llm.to(self._device)

        self._dtype = dtype
        self._max_passage_words = max_passage_words

    def _compute_effective_max_words(self, num_docs: int) -> int | None:
        """Derive a per-document word limit when ``max_passage_words`` is None.

        Uses a rough 1 token ~= 0.75 words heuristic.  Reserves ~128 tokens
        for the query and structural overhead per document.
        """
        if self._max_passage_words is not None:
            return self._max_passage_words
        query_reserve_tokens = 128
        available_tokens = self._context_size - query_reserve_tokens
        tokens_per_doc = available_tokens // max(num_docs, 1)
        words_per_doc = int(tokens_per_doc * 0.75)
        return max(words_per_doc, 16)

    def _compute_docs_per_chunk(self, avg_doc_words: int) -> int:
        """Estimate how many documents fit in one forward pass.

        When ``max_passage_words`` is set we can calculate roughly how many
        documents fit.  The result is capped at ``self._batch_size``
        (which itself is <= 64).
        """
        tokens_per_word = 1.34  # conservative estimate
        query_reserve_tokens = 128
        per_doc_tokens = int(avg_doc_words * tokens_per_word) + 32  # overhead per doc
        available_tokens = self._context_size - query_reserve_tokens
        fit = max(available_tokens // max(per_doc_tokens, 1), 1)
        return min(fit, self._batch_size)

    @staticmethod
    def _normalise_scores(scores: list[float]) -> list[float]:
        """Min-max normalise a list of scores to [0, 1]."""
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if math.isclose(lo, hi):
            return [0.5] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

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

        total_candidates = sum(len(r.candidates) for r in rerank_results)

        with tqdm(
            total=total_candidates, desc="Jina reranking (q, d) pairs"
        ) as progress_bar:
            for result in rerank_results:
                query_text = result.query.text
                candidates = result.candidates
                num_candidates = len(candidates)

                effective_max_words = self._compute_effective_max_words(
                    min(num_candidates, self._batch_size)
                )
                doc_texts = [
                    _extract_doc_text(c.doc, max_words=effective_max_words)
                    for c in candidates
                ]

                if self._max_passage_words is not None:
                    avg_words = (
                        sum(len(t.split()) for t in doc_texts) / max(len(doc_texts), 1)
                    )
                    chunk_size = self._compute_docs_per_chunk(int(avg_words))
                else:
                    chunk_size = self._batch_size

                needs_normalisation = num_candidates > chunk_size
                all_scores: list[float] = [0.0] * num_candidates

                for chunk_start in range(0, num_candidates, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_candidates)
                    chunk_docs = doc_texts[chunk_start:chunk_end]

                    jina_results = self._llm.rerank(query_text, chunk_docs)

                    chunk_scores: list[float] = [0.0] * len(chunk_docs)
                    for item in jina_results:
                        chunk_scores[item["index"]] = item["relevance_score"]

                    if needs_normalisation:
                        chunk_scores = self._normalise_scores(chunk_scores)

                    for i, score in enumerate(chunk_scores):
                        all_scores[chunk_start + i] = score

                    if populate_invocations_history:
                        prompt_repr = (
                            f"query={query_text!r}, "
                            f"docs[{chunk_start}:{chunk_end}] "
                            f"({len(chunk_docs)} docs)"
                        )
                        response_repr = ", ".join(
                            f"[{chunk_start + i}]={s:.4f}"
                            for i, s in enumerate(chunk_scores)
                        )
                        result.invocations_history.append(
                            InferenceInvocation(
                                prompt=prompt_repr,
                                response=response_repr,
                                input_token_count=0,
                                output_token_count=0,
                            )
                        )

                    progress_bar.update(len(chunk_docs))

                for idx, score in enumerate(all_scores):
                    result.candidates[idx].score = score

                result.candidates.sort(
                    key=cmp_to_key(self.candidate_comparator), reverse=True
                )

        return rerank_results

    # -- Abstract method implementations (not used in primary rerank path) ----

    def run_llm_batched(
        self,
        prompts: list[str],
    ) -> tuple[list[str], list[int], list[float]]:
        raise NotImplementedError(
            "JinaReranker uses model.rerank() directly; "
            "see rerank_batch() instead."
        )

    def run_llm(self, prompt: str) -> tuple[str, int, float]:
        raise NotImplementedError(
            "JinaReranker uses model.rerank() directly; "
            "see rerank_batch() instead."
        )

    def create_prompt(self, result: Result, index: int) -> tuple[str, int]:
        query = result.query.text
        doc_text = _extract_doc_text(
            result.candidates[index].doc,
            max_words=self._max_passage_words,
        )
        prompt = f"Query: {query} Document: {doc_text}"
        return prompt, len(prompt.split())

    def get_num_tokens(self, prompt: str) -> int:
        return int(len(prompt.split()) * 1.34)

    def num_output_tokens(self) -> int:
        return 0

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

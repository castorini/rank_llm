"""Pointwise yes/no reranking via an OpenAI-compatible vLLM server (logprob scoring)."""

from __future__ import annotations

import asyncio
import copy
import logging
from functools import cmp_to_key
from importlib.resources import files
from typing import Any

from tqdm import tqdm

from rank_llm.data import InferenceInvocation, Request, Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.vllm_handler_with_openai_sdk import (
    POINTWISE_NO_LOGPROB_TOKEN_STRINGS,
    POINTWISE_YES_LOGPROB_TOKEN_STRINGS,
    VllmHandlerWithOpenAISDK,
)

logger = logging.getLogger(__name__)

TEMPLATES = files("rank_llm.rerank.prompt_templates")


def _unique_token_ids_from_strings(tokenizer: Any, strings: tuple[str, ...]) -> list[int]:
    """Token ids for constrained decoding; deduped, order preserved."""
    seen: set[int] = set()
    out: list[int] = []
    for s in strings:
        for tid in tokenizer.encode(s, add_special_tokens=False):
            if tid not in seen:
                seen.add(tid)
                out.append(tid)
    return out


class PointwiseVLLM(PointwiseRankLLM):
    """
    Pointwise relevance with a single-token yes/no style output and
    logprob-based scores over an OpenAI-compatible vLLM HTTP API.

    This is a generative pointwise judge (prompt + one constrained token), not
    a small neural cross-encoder like MonoT5. Any chat LLM served with vLLM can
    work if it follows the template and ``allowed_token_ids``; Qwen, Llama,
    Mistral, etc. differ mainly by tokenizer and optional chat extras (e.g.
    ``enable_thinking``).

    Few-shot examples are not supported (``num_few_shot_examples`` is fixed at 0).

    Sync ``rerank_batch`` runs HTTP calls concurrently per micro-batch.
    Async ``rerank_batch_async`` uses a per-instance semaphore so overlapping
    calls share one concurrency cap.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        prompt_mode: PromptMode | None = None,
        prompt_template_path: str = (TEMPLATES / "pointwise_vllm_template.yaml"),
        context_size: int = 8192,
        batch_size: int = 32,
        max_concurrent_llm_calls: int | None = None,
        disable_thinking_extra_body: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=0,
            few_shot_file=None,
            device="remote",
            batch_size=batch_size,
        )
        self._vllm = VllmHandlerWithOpenAISDK(base_url=base_url, model=model)
        self._tokenizer = self._vllm.get_tokenizer()
        self._llm_concurrency_sem: asyncio.Semaphore | None = None
        self._max_concurrent = max_concurrent_llm_calls or max(batch_size, 1)

        yes_ids = _unique_token_ids_from_strings(
            self._tokenizer, POINTWISE_YES_LOGPROB_TOKEN_STRINGS
        )
        no_ids = _unique_token_ids_from_strings(
            self._tokenizer, POINTWISE_NO_LOGPROB_TOKEN_STRINGS
        )
        self._allowed_token_ids = list(dict.fromkeys(yes_ids + no_ids))

        self._score_extra_body: dict[str, Any] = {
            "allowed_token_ids": self._allowed_token_ids,
        }
        if disable_thinking_extra_body:
            self._score_extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    def _get_llm_concurrency_sem(self) -> asyncio.Semaphore:
        if self._llm_concurrency_sem is None:
            self._llm_concurrency_sem = asyncio.Semaphore(self._max_concurrent)
        return self._llm_concurrency_sem

    def _input_token_count(self, messages: list[dict[str, str]]) -> int:
        ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        return len(ids)

    def _probe_messages(self, result: Result) -> list[dict[str, str]]:
        """Messages matching ``generate_chat_messages`` layout with an empty document."""
        query = self._inference_handler._replace_number(result.query.text)
        body_empty = self._inference_handler._format_template(
            "body",
            {"query": query, "doc_content": ""},
        )
        user_content = body_empty.replace("<unk>", "")
        messages: list[dict[str, str]] = []
        system_msg = self._inference_handler.template.get("system_message")
        if system_msg:
            messages.append({"role": "system", "content": system_msg.strip()})
        messages.append({"role": "user", "content": user_content})
        return messages

    def create_prompt(self, result: Result, index: int) -> tuple[list[dict[str, str]], int]:
        reserved_for_output = 8
        overhead = self._input_token_count(self._probe_messages(result))
        max_doc_tokens = max(
            16,
            self._context_size - overhead - reserved_for_output,
        )

        messages = self._inference_handler.generate_chat_messages(
            result=result,
            index=index,
            max_doc_tokens=max_doc_tokens,
            tokenizer=self._tokenizer,
            num_few_shot_examples=0,
            fewshot_examples=[],
        )
        n_tok = self._input_token_count(messages)
        if n_tok > self._context_size - reserved_for_output:
            logger.warning(
                "Prompt length %s exceeds budget %s; consider lowering rank_end or context.",
                n_tok,
                self._context_size - reserved_for_output,
            )
        return messages, n_tok

    def get_num_tokens(self, prompt: str | list[dict[str, str]]) -> int:
        if isinstance(prompt, list):
            return self._input_token_count(prompt)
        return len(self._tokenizer.encode(prompt))

    def num_output_tokens(self) -> int:
        return 1

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0.0

    async def _score_one_async(
        self, messages: list[dict[str, str]], *, use_sem: bool
    ) -> tuple[str, int, float, dict[str, Any]]:
        if use_sem:
            async with self._get_llm_concurrency_sem():
                return await self._vllm.chat_completion_score_async(
                    messages,
                    extra_body=self._score_extra_body,
                )
        return await self._vllm.chat_completion_score_async(
            messages,
            extra_body=self._score_extra_body,
        )

    def run_llm(
        self, prompt: str | list[dict[str, str]], **kwargs: Any
    ) -> tuple[str, int, float]:
        if not isinstance(prompt, list):
            raise TypeError("PointwiseVLLM expects chat messages (list of dicts).")

        async def _one() -> tuple[str, int, float, dict[str, Any]]:
            return await self._score_one_async(prompt, use_sem=False)

        text, out_tok, score, _usage = asyncio.run(_one())
        return text, out_tok, score

    def run_llm_batched(
        self,
        prompts: list[str | list[dict[str, str]]],
        **kwargs: Any,
    ) -> tuple[list[str], list[int], list[float]]:
        messages_list = [p for p in prompts if isinstance(p, list)]
        if len(messages_list) != len(prompts):
            raise TypeError("PointwiseVLLM batch expects chat message lists.")

        async def _all() -> list[tuple[str, int, float, dict[str, Any]]]:
            return await asyncio.gather(
                *[self._score_one_async(m, use_sem=False) for m in messages_list]
            )

        rows = asyncio.run(_all())
        texts, out_counts, scores = [], [], []
        for text, out_tok, score, _u in rows:
            texts.append(text)
            out_counts.append(out_tok)
            scores.append(score)
        return texts, out_counts, scores

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

        for res in rerank_results:
            if rank_end > 0:
                res.candidates = res.candidates[rank_start:rank_end]

        total_candidates = sum(len(r.candidates) for r in rerank_results)

        with tqdm(
            total=total_candidates, desc="Progress through (q, d) pairs"
        ) as progress_bar:
            index = 0
            while index < total_candidates:
                prompts, token_counts = self.create_prompt_batched(
                    results=rerank_results, index=index
                )

                outputs, output_token_counts, scores = self.run_llm_batched(
                    prompts=prompts
                )

                for i, score in enumerate(scores):
                    qn, cn = self.get_query_and_candidate_index(rerank_results, index + i)
                    rerank_results[qn].candidates[cn].score = score
                    if populate_invocations_history:
                        rerank_results[qn].invocations_history.append(
                            InferenceInvocation(
                                prompts[i],
                                outputs[i],
                                token_counts[i],
                                output_token_counts[i],
                            )
                        )

                progress_bar.update(len(scores))
                index += self._batch_size

        for result in rerank_results:
            result.candidates.sort(
                key=cmp_to_key(self.candidate_comparator), reverse=True
            )

        return rerank_results

    async def rerank_batch_async(
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

        for res in rerank_results:
            if rank_end > 0:
                res.candidates = res.candidates[rank_start:rank_end]

        work: list[tuple[int, int, list[dict[str, str]], int]] = []
        for qi, res in enumerate(rerank_results):
            for ci in range(len(res.candidates)):
                msgs, n_tok = self.create_prompt(res, ci)
                work.append((qi, ci, msgs, n_tok))

        sem = self._get_llm_concurrency_sem()

        async def score_one(
            qi: int, ci: int, msgs: list[dict[str, str]], n_tok: int
        ) -> tuple[int, int, list[dict[str, str]], int, str, int, float, dict[str, Any]]:
            async with sem:
                text, out_tok, score, usage = (
                    await self._vllm.chat_completion_score_async(
                        msgs,
                        extra_body=self._score_extra_body,
                    )
                )
            return qi, ci, msgs, n_tok, text, out_tok, score, usage

        rows = await asyncio.gather(
            *[score_one(qi, ci, msgs, n_tok) for qi, ci, msgs, n_tok in work]
        )

        for qi, ci, msgs, n_tok, text, out_tok, score, usage in rows:
            rerank_results[qi].candidates[ci].score = score
            if populate_invocations_history:
                in_tok = int(
                    usage.get("prompt_tokens")
                    or usage.get("input_tokens")
                    or n_tok
                )
                rerank_results[qi].invocations_history.append(
                    InferenceInvocation(
                        msgs,
                        text,
                        in_tok,
                        out_tok,
                        token_usage=usage,
                    )
                )

        for result in rerank_results:
            result.candidates.sort(
                key=cmp_to_key(self.candidate_comparator), reverse=True
            )

        return rerank_results

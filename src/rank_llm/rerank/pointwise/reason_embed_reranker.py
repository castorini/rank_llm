import asyncio
import copy
import logging
import os
import re
from importlib.resources import files
from typing import Any

from ftfy import fix_text
from tqdm import tqdm

from rank_llm.data import Request, Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.vllm_handler import VllmHandler

logger = logging.getLogger(__name__)
TEMPLATES = files("rank_llm.rerank.prompt_templates")


class ReasonEmbedReranker(PointwiseRankLLM):
    def __init__(
        self,
        model_path: str,
        prompt_mode: PromptMode = None,
        prompt_template_path: str = (TEMPLATES / "reason_embed_template.yaml"),
        context_size: int = 40960,
        num_few_shot_examples: int = 0,
        few_shot_file: str = None,
        device: str = "cuda",
        batch_size: int = 1,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = -1,
        do_sample: bool = True,
        logprobs: int = 10,
        *,
        relevance_definition: str,
        query_type: str = "user query",
        doc_type: str = "retrieved document",
    ):
        super().__init__(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            device=device,
            batch_size=batch_size,
        )
        logger.info(
            "ReasonEmbedReranker: using attn_implementation='%s'",
            attn_implementation,
        )

        self._vllm_handler = VllmHandler(
            model=model_path,
            download_dir=os.getenv("HF_HOME"),
            enforce_eager=False,
            max_logprobs=logprobs,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        self._tokenizer = self._vllm_handler.get_tokenizer()

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._do_sample = do_sample
        self._logprobs = logprobs
        self._relevance_definition = relevance_definition
        self._query_type = query_type
        self._doc_type = doc_type

        self._materialize_template_placeholders()

        self._system_prompt = self._inference_handler.template.get(
            "system_message",
            "You are an expert in retrieval relevance assessment. "
            "Reason carefully and output exactly one final relevance score in <score> tags.",
        )

    def _extract_doc_text(self, doc: Any) -> str:
        if not isinstance(doc, dict):
            return str(doc)
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
        else:
            content = doc.get("passage", "")
        if doc.get("title"):
            content = f"Title: {doc['title']} Content: {content}"
        return fix_text(str(content).strip())

    def _build_user_input(self, query: str, doc_text: str) -> str:
        fmt_values = {"query": query, "doc_content": doc_text}
        return self._inference_handler._format_template(
            template_key="body", fmt_values=fmt_values
        )

    def _materialize_template_placeholders(self) -> None:
        replacements = {
            "{relevance_definition}": self._escape_template_value(
                self._relevance_definition
            ),
            "{query_type}": self._escape_template_value(self._query_type),
            "{doc_type}": self._escape_template_value(self._doc_type),
        }

        for key in ["system_message", "body"]:
            template_text = self._inference_handler.template.get(key)
            if not template_text:
                continue
            for placeholder, value in replacements.items():
                template_text = template_text.replace(placeholder, value)
            self._inference_handler.template[key] = template_text

    def _escape_template_value(self, value: str) -> str:
        return str(value).replace("{", "{{").replace("}", "}}")

    def _truncate_doc_text(self, query: str, doc_text: str) -> str:
        query_tokens = len(self._tokenizer.encode(query, add_special_tokens=False))
        max_doc_tokens = max(self._context_size - query_tokens - 512, 128)
        doc_tokens = self._tokenizer.encode(doc_text, add_special_tokens=False)
        if len(doc_tokens) <= max_doc_tokens:
            return doc_text
        return self._tokenizer.decode(
            doc_tokens[:max_doc_tokens], skip_special_tokens=True
        )

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _parse_score(self, response: str) -> float:
        match = re.search(r"<score>\s*(\d{1,3})\s*</score>", response)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 100.0)

        # Fallback: extract first standalone integer and clamp.
        fallback = re.search(r"\b(\d{1,3})\b", response)
        if fallback:
            score = float(fallback.group(1))
            return min(max(score, 0.0), 100.0)
        return 0.0

    def _score_candidate(self, query: str, doc_text: str) -> float:
        doc_text = self._truncate_doc_text(query, doc_text)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._build_user_input(query, doc_text)},
        ]
        text_input = self._apply_chat_template(messages)
        response, _, _ = asyncio.run(
            self._vllm_handler.generate_output_async(
                prompt=text_input,
                min_tokens=1,
                max_tokens=self._max_new_tokens,
                temperature=self._temperature if self._do_sample else 0.0,
                top_p=self._top_p,
                top_k=self._top_k,
                logprobs=self._logprobs,
            )
        )
        return self._parse_score(response)

    def rerank_batch(
        self,
        requests: list[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> list[Result]:
        results = [
            Result(
                query=copy.deepcopy(r.query),
                candidates=copy.deepcopy(r.candidates),
                invocations_history=[],
            )
            for r in requests
        ]
        total = sum(len(res.candidates[rank_start:rank_end]) for res in results)
        with tqdm(total=total, desc="ReasonEmbed reranking") as progress:
            for result in results:
                query = result.query.text
                for cand in result.candidates[rank_start:rank_end]:
                    doc_text = self._extract_doc_text(cand.doc)
                    cand.score = self._score_candidate(query, doc_text)
                    progress.update(1)
                result.candidates.sort(key=lambda x: x.score, reverse=True)
        return results

    # --- Abstract method stubs (rerank_batch is overridden above) ---

    def run_llm(self, prompt: str | list[dict[str, str]], **kwargs) -> tuple[str, int]:
        raise NotImplementedError("Use rerank_batch directly.")

    def run_llm_batched(
        self, prompts: list[str | list[dict[str, str]]], **kwargs
    ) -> list[tuple[str, int]]:
        raise NotImplementedError("Use rerank_batch directly.")

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> tuple[str, int]:
        raise NotImplementedError("Use rerank_batch directly.")

    def get_num_tokens(self, prompt: str | list[dict[str, str]]) -> int:
        if isinstance(prompt, str):
            return len(self._tokenizer.encode(prompt))
        return sum(len(self._tokenizer.encode(m["content"])) for m in prompt)

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0.0

    def num_output_tokens(self) -> int:
        return self._max_new_tokens

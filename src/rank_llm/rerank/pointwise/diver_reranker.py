import asyncio
import copy
import json
import logging
import os
from importlib.resources import files
from typing import Any

from ftfy import fix_text
from tqdm import tqdm
from transformers import AutoTokenizer

from rank_llm.data import Request, Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.vllm_handler import VllmHandler

logger = logging.getLogger(__name__)
TEMPLATES = files("rank_llm.rerank.prompt_templates")


class DiverPointwiseReranker(PointwiseRankLLM):
    def __init__(
        self,
        model_path: str,
        prompt_mode: PromptMode = None,
        prompt_template_path: str = (TEMPLATES / "diver_template.yaml"),
        context_size: int = 32768,
        num_few_shot_examples: int = 0,
        few_shot_file: str = None,
        device: str = "cuda",
        batch_size: int = 1,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 8000,
        do_sample: bool = True,
        temperature: float = 0.3,
        top_p: float = 0.8,
        top_k: int = -1,
        repetition_penalty: float = 1.05,
        logprobs: int = 10,
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
            "DiverPointwiseReranker: using attn_implementation='%s'",
            attn_implementation,
        )

        self._vllm_handler = VllmHandler(
            model=model_path,
            download_dir=os.getenv("HF_HOME"),
            enforce_eager=False,
            max_logprobs=logprobs,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=32000,
            trust_remote_code=True,
        )
        self._tokenizer = self._load_tokenizer(model_path)
        if not getattr(self._tokenizer, "chat_template", None):
            # Qwen-style ChatML template (matches Qwen2/3 default formatting).
            self._tokenizer.chat_template = (
                "{% if not add_generation_prompt is defined %}"
                "{% set add_generation_prompt = false %}"
                "{% endif %}"
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'user' %}"
                "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            )
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._repetition_penalty = repetition_penalty
        self._logprobs = logprobs

        self._system_prompt = self._inference_handler.template.get(
            "system_message",
            "Your task is to evaluate and rank documents based on how well they help "
            "answer the given query. Prioritize usefulness/helpfulness first, then relevance.",
        )

    def _load_tokenizer(self, model_path: str):
        try:
            return self._vllm_handler.get_tokenizer()
        except ValueError as err:
            if "a coroutine was expected" not in str(err):
                raise
            logger.warning(
                "Falling back to AutoTokenizer because vLLM returned a synchronous tokenizer object."
            )
            return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _build_user_input(self, query: str, doc_text: str) -> str:
        fmt_values = {"query": query, "doc_content": doc_text}
        return self._inference_handler._format_template(
            template_key="body", fmt_values=fmt_values
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

    def _truncate_doc_text(self, query: str, doc_text: str) -> str:
        # Keep a safety margin for system + prompt overhead and query tokens.
        query_tokens = self.get_num_tokens(query)
        max_doc_tokens = max(
            self._context_size - self._max_new_tokens - query_tokens - 256, 128
        )
        doc_tokens = self._tokenizer.encode(doc_text, add_special_tokens=False)
        if len(doc_tokens) <= max_doc_tokens:
            return doc_text
        doc_tokens = doc_tokens[:max_doc_tokens]
        return self._tokenizer.decode(doc_tokens, skip_special_tokens=True)

    def _prepare_prompt(self, query: str, doc_text: str) -> str:
        doc_text = self._truncate_doc_text(query, doc_text)
        prompt_budget = max(self._context_size - self._max_new_tokens, 128)

        while True:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._build_user_input(query, doc_text)},
            ]
            if not getattr(self._tokenizer, "chat_template", None):
                raise ValueError(
                    "Tokenizer chat_template is missing for Diver; set a Qwen ChatML template explicitly."
                )
            text_input = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = len(
                self._tokenizer.encode(text_input, add_special_tokens=False)
            )
            if prompt_tokens <= prompt_budget:
                return text_input

            doc_tokens = self._tokenizer.encode(doc_text, add_special_tokens=False)
            overflow = prompt_tokens - prompt_budget
            next_doc_len = max(len(doc_tokens) - overflow - 64, 128)
            if next_doc_len >= len(doc_tokens):
                next_doc_len = max(len(doc_tokens) - 128, 128)
            doc_text = self._tokenizer.decode(
                doc_tokens[:next_doc_len], skip_special_tokens=True
            )

    def _parse_score(self, response: str) -> float:
        # 5. Parsing logic
        try:
            text = response.strip()
            # Preferred format from template: raw JSON only.
            if text.startswith("{") and text.endswith("}"):
                scores = json.loads(text)
                return float(scores.get("[1]", scores.get("1", 0.0)))

            # Backward compatibility for older fenced-json outputs.
            if "```json" in response:
                json_part = response.split("```json", 1)[1].split("```", 1)[0].strip()
                scores = json.loads(json_part)
                return float(scores.get("[1]", scores.get("1", 0.0)))

            # Last-resort extraction if model adds extra text around a JSON object.
            left = response.find("{")
            right = response.rfind("}")
            if left != -1 and right != -1 and right > left:
                scores = json.loads(response[left : right + 1])
                return float(scores.get("[1]", scores.get("1", 0.0)))
        except Exception:
            pass
        return 0.0

    async def _run_prompts_async(
        self, prompts: list[str]
    ) -> list[tuple[str, int, int]]:
        return await asyncio.gather(
            *[
                self._vllm_handler.generate_output_async(
                    prompt=prompt,
                    min_tokens=1,
                    max_tokens=self._max_new_tokens,
                    temperature=self._temperature if self._do_sample else 0.0,
                    top_p=self._top_p,
                    top_k=self._top_k,
                    logprobs=self._logprobs,
                )
                for prompt in prompts
            ]
        )

    async def _score_candidates_batched_async(
        self, query: str, doc_texts: list[str]
    ) -> list[float]:
        prompts = [self._prepare_prompt(query, doc_text) for doc_text in doc_texts]
        responses = await self._run_prompts_async(prompts)
        return [self._parse_score(response) for response, _, _ in responses]

    async def _rerank_results_async(
        self,
        results: list[Result],
        rank_start: int,
        rank_end: int,
    ) -> None:
        total = sum(len(res.candidates[rank_start:rank_end]) for res in results)
        with tqdm(total=total, desc="Diver reranking") as progress:
            for result in results:
                query = result.query.text
                candidates = result.candidates[rank_start:rank_end]
                for batch_start in range(0, len(candidates), self._batch_size):
                    batch = candidates[batch_start : batch_start + self._batch_size]
                    doc_texts = [self._extract_doc_text(cand.doc) for cand in batch]
                    scores = await self._score_candidates_batched_async(
                        query, doc_texts
                    )
                    for cand, score in zip(batch, scores):
                        cand.score = score
                    progress.update(len(batch))
                result.candidates.sort(key=lambda x: x.score, reverse=True)

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
        asyncio.run(self._rerank_results_async(results, rank_start, rank_end))
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

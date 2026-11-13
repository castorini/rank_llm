import logging
from importlib.resources import files
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM

logger = logging.getLogger(__name__)

TEMPLATES = files("rank_llm.rerank.prompt_templates")


class CrossEncoderReranker(PointwiseRankLLM):
    """
    Pointwise Neural Cross-Encoder Reranker using Hugging Face AutoModelForSequenceClassification.

    Compatible with standard cross-encoder architectures, including:
        - BAAI/bge-reranker-base / BAAI/bge-reranker-large / BAAI/bge-reranker-v2-m3
        - cross-encoder/ms-marco-MiniLM-L-6-v2 / cross-encoder/ms-marco-TinyBERT-L-2-v2
        - jinaai/jina-reranker-v1-base-en / jinaai/jina-reranker-v2-base-multilingual
        - Any sequence classification model trained on query-passage pairs.

    Handles single-logit scoring, two-class classification (class 1 as relevant relative to class 0),
    and optional sigmoid probability normalization for regression checkpoints.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        prompt_template_path: Any | None = None,
        context_size: int = 512,
        device: str | None = None,
        batch_size: int = 32,
        apply_sigmoid: bool = False,
        trust_remote_code: bool = False,
    ):
        default_template = TEMPLATES / "cross_encoder_template.yaml"
        if prompt_template_path is None:
            prompt_template_path = default_template
        elif str(prompt_template_path) != str(default_template):
            raise ValueError(
                f"CrossEncoderReranker operates directly on query-document pairs via sequence "
                f"classification tokenization and does not support custom prompt templates ({prompt_template_path})."
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

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

        self._apply_sigmoid = apply_sigmoid
        self._tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
        self._llm = AutoModelForSequenceClassification.from_pretrained(
            model, trust_remote_code=trust_remote_code
        ).to(self._device)
        self._llm.eval()

        # Clamp context_size to the model's physical positional limit
        model_limit = None
        if hasattr(self._llm, "config"):
            cfg_limit = getattr(self._llm.config, "max_position_embeddings", None)
            if isinstance(cfg_limit, int):
                model_limit = cfg_limit

        if model_limit is None and hasattr(self._tokenizer, "model_max_length"):
            tok_limit = getattr(self._tokenizer, "model_max_length", None)
            if isinstance(tok_limit, int) and tok_limit < 100000:
                model_limit = tok_limit

        if isinstance(model_limit, int) and model_limit > 0:
            if context_size > model_limit:
                logger.info(
                    "Clamping context_size from %s to model max limit %s",
                    context_size,
                    model_limit,
                )
            self._context_size = min(context_size, model_limit)
        else:
            self._context_size = context_size

    def run_llm_batched(
        self,
        prompts: list[str],
    ) -> tuple[list[str], list[int], list[float]]:
        """
        Run batched inference across query-document pairs.

        Args:
            prompts: List of formatted query-document strings.

        Returns:
            Tuple of (outputs, token_counts, scores)
        """
        all_outputs: list[str] = []
        all_output_token_counts: list[int] = []
        all_scores: list[float] = []

        queries: list[str] = []
        texts: list[str] = []
        for prompt in prompts:
            query, text = self._parse_prompt(prompt)
            queries.append(query)
            texts.append(text)

        with torch.no_grad():
            inputs = self._tokenizer(
                queries,
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._context_size,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            outputs = self._llm(**inputs)
            logits = outputs.logits

            # Determine scoring format based on tensor shape:
            # Case 1: Single output logit per item (shape: [batch_size] or [batch_size, 1])
            if logits.ndim == 1:
                raw_scores = logits
                if self._apply_sigmoid:
                    raw_scores = torch.sigmoid(raw_scores)
            elif logits.shape[1] == 1:
                raw_scores = logits.squeeze(-1)
                if self._apply_sigmoid:
                    raw_scores = torch.sigmoid(raw_scores)
            # Case 2: Binary classification logits (shape: [batch_size, 2])
            elif logits.shape[1] == 2:
                # Class 1 represents relevant, class 0 represents non-relevant.
                # Compute relative relevance via softmax probability for class 1
                scores_2d = torch.softmax(logits, dim=-1)
                raw_scores = scores_2d[:, 1]
            else:
                # Multi-class classification: softmax over classes and take final class
                scores_nd = torch.softmax(logits, dim=-1)
                raw_scores = scores_nd[:, -1]

            numpy_scores = raw_scores.detach().cpu().to(torch.float32).numpy()

            for score in numpy_scores:
                float_score = float(score)
                all_scores.append(float_score)
                all_output_token_counts.append(0)
                all_outputs.append(str(float_score))

        return all_outputs, all_output_token_counts, all_scores

    def run_llm(self, prompt: str) -> tuple[str, int, float]:
        """Run single inference."""
        ret = self.run_llm_batched([prompt])
        return ret[0][0], ret[1][0], ret[2][0]

    def create_prompt(self, result: Result, index: int) -> tuple[str, int]:
        """Format query and candidate document into paired representation."""
        query = self._inference_handler._replace_number(result.query.text)
        candidate = result.candidates[index]

        if isinstance(candidate.doc, dict):
            doc_text = self._inference_handler._convert_doc_to_prompt_content(
                candidate.doc, max_length=self._max_passage_words
            )
        else:
            doc_text = str(candidate.doc)

        query_tokens = self.get_num_tokens(query)
        max_doc_tokens = max(10, self._context_size - query_tokens - 10)

        if self.get_num_tokens(doc_text) > max_doc_tokens:
            doc_tokens = self._tokenizer.encode(doc_text, add_special_tokens=False)
            doc_tokens = doc_tokens[:max_doc_tokens]
            doc_text = self._tokenizer.decode(doc_tokens, skip_special_tokens=True)

        prompt = f"QUERY: {query} DOCUMENT: {doc_text}"
        final_token_count = self.get_num_tokens(prompt)
        return prompt, final_token_count

    def _parse_prompt(self, prompt: str) -> tuple[str, str]:
        """Parse query and document from prompt string."""
        parts = prompt.split(" DOCUMENT: ", 1)
        if len(parts) == 2:
            query = parts[0].replace("QUERY: ", "")
            document = parts[1]
            return query, document
        return prompt, ""

    def get_num_tokens(self, text: str) -> int:
        """Get number of tokens for text."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def num_output_tokens(self) -> int:
        """Sequence classification model produces no output text tokens."""
        return 0

    def cost_per_1k_token(self, input_token: bool) -> float:
        """Local weights, zero API cost."""
        return 0.0

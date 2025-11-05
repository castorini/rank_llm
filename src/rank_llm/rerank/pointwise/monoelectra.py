import logging
from importlib.resources import files
from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM

logger = logging.getLogger(__name__)

TEMPLATES = files("rank_llm.rerank.prompt_templates")


class MonoELECTRA(PointwiseRankLLM):
    def __init__(
        self,
        model: str = "castorini/monoelectra-base",
        prompt_template_path: str = (TEMPLATES / "monoelectra_template.yaml"),
        context_size: int = 512,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        # ELECTRA doesn't use prompt templates in practice, but we need a minimal one
        # for the base class initialization
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

        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._llm = AutoModelForSequenceClassification.from_pretrained(model).to(
            self._device
        )
        self._llm.eval()
        self._context_size = context_size

    def run_llm_batched(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Run batched inference on ELECTRA model.

        Args:
            prompts: List of query-document pairs formatted as strings

        Returns:
            Tuple of (outputs, token_counts, scores)
        """
        all_outputs = []
        all_output_token_counts = []
        all_scores = []

        # Parse query-document pairs from prompts
        queries = []
        texts = []
        for prompt in prompts:
            # prompts are stored as formatted strings containing query and document
            # We need to extract them
            query, text = self._parse_prompt(prompt)
            queries.append(query)
            texts.append(text)

        with torch.no_grad():
            # Tokenize query-document pairs
            inputs = self._tokenizer(
                queries,
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._context_size,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Get model predictions
            outputs = self._llm(**inputs)

            # Extract relevance scores (logits for class 1 = relevant)
            logits = outputs.logits[:, 1].cpu().detach().numpy()

            for score in logits:
                all_scores.append(float(score))
                # For ELECTRA, output is just a single logit, no tokens generated
                all_output_token_counts.append(0)
                all_outputs.append(str(score))

        return all_outputs, all_output_token_counts, all_scores

    def run_llm(self, prompt: str) -> Tuple[str, int, float]:
        """Run single inference."""
        ret = self.run_llm_batched([prompt])
        return ret[0][0], ret[1][0], ret[2][0]

    def create_prompt(self, result: Result, index: int) -> Tuple[str, int]:
        """
        Create a prompt for ELECTRA.

        Since ELECTRA doesn't use a template, we store the query and document
        in a simple format that can be parsed later.
        """
        query = result.query.text
        candidate = result.candidates[index]

        # Get document text - check common field names
        doc_text = ""
        if "text" in candidate.doc:
            doc_text = candidate.doc["text"]
        elif "contents" in candidate.doc:
            doc_text = candidate.doc["contents"]
        elif "content" in candidate.doc:
            doc_text = candidate.doc["content"]
        elif "body" in candidate.doc:
            doc_text = candidate.doc["body"]
        else:
            # Fallback: use string representation
            doc_text = str(candidate.doc)

        # Truncate document if needed
        # We need to account for query + document + special tokens
        query_tokens = self.get_num_tokens(query)
        max_doc_tokens = (
            self._context_size - query_tokens - 10
        )  # Reserve for special tokens

        if self.get_num_tokens(doc_text) > max_doc_tokens:
            doc_tokens = self._tokenizer.encode(doc_text, add_special_tokens=False)
            doc_tokens = doc_tokens[:max_doc_tokens]
            doc_text = self._tokenizer.decode(doc_tokens, skip_special_tokens=True)

        # Store as a simple format: "QUERY: <query> DOCUMENT: <doc>"
        # This will be parsed in run_llm_batched
        prompt = f"QUERY: {query} DOCUMENT: {doc_text}"

        # Calculate final token count
        final_token_count = self.get_num_tokens(prompt)

        return prompt, final_token_count

    def _parse_prompt(self, prompt: str) -> Tuple[str, str]:
        """Parse query and document from prompt string."""
        parts = prompt.split(" DOCUMENT: ", 1)
        if len(parts) == 2:
            query = parts[0].replace("QUERY: ", "")
            document = parts[1]
            return query, document
        else:
            # Fallback
            return prompt, ""

    def get_num_tokens(self, text: str) -> int:
        """Get number of tokens for a text string."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def num_output_tokens(self) -> int:
        """ELECTRA is encoder-only, produces no output tokens."""
        return 0

    def cost_per_1k_token(self, input_token: bool) -> float:
        """Open source model, no cost."""
        return 0

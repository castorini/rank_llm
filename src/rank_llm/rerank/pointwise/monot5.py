import logging
import math
from importlib.resources import files
from typing import List, Optional, Tuple

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from transformers.generation import GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    T5ForConditionalGeneration = None
    T5Tokenizer = None
    GenerationConfig = None

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM
from rank_llm.rerank.rankllm import PromptMode

logger = logging.getLogger(__name__)

TEMPLATES = files("rank_llm.rerank.prompt_templates")


class MonoT5(PointwiseRankLLM):
    def __init__(
        self,
        model: str,
        prompt_mode: Optional[PromptMode] = None,
        prompt_template_path: str = (TEMPLATES / "monot5_template.yaml"),
        context_size: int = 512,
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. Please install it with: pip install rank_llm[transformers]"
            )
        
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            device=device,
            batch_size=batch_size,
        )

        self._tokenizer = T5Tokenizer.from_pretrained(model)
        self._llm = T5ForConditionalGeneration.from_pretrained(model).to(self._device)
        self._context_size = context_size

        self._true_id = self._tokenizer.encode("true", add_special_tokens=False)[0]
        self._false_id = self._tokenizer.encode("false", add_special_tokens=False)[0]

    def run_llm_batched(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], List[int], List[float]]:
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens()
        gen_cfg.min_new_tokens = self.num_output_tokens()
        gen_cfg.output_scores = True
        gen_cfg.return_dict_in_generate = True
        gen_cfg.do_sample = False
        all_outputs, all_output_token_counts, all_scores = [], [], []

        token_prompts = self._tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(self._device)

        token_prompts = token_prompts["input_ids"]

        batch_outputs = self._llm.generate(token_prompts, generation_config=gen_cfg)

        batch_output_ids = batch_outputs.sequences
        batch_logits = batch_outputs.scores

        batch_outputs = [
            self._tokenizer.decode(
                single_token_sequence,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            for single_token_sequence in batch_output_ids
        ]

        for logit_tensor in batch_logits[0]:
            truth_logit = logit_tensor[self._true_id]
            false_logit = logit_tensor[self._false_id]
            score = math.exp(truth_logit) / (
                math.exp(truth_logit) + math.exp(false_logit)
            )
            all_scores.append(score)
            all_output_token_counts.append(self.num_output_tokens())

        all_outputs.extend(batch_outputs)

        return all_outputs, all_output_token_counts, all_scores

    def run_llm(self, prompt: str) -> Tuple[str, int, float]:
        ret = self.run_llm_batched([prompt])
        return ret[0][0], ret[1][0], ret[2][0]

    def create_prompt(self, result: Result, index: int) -> Tuple[str, int]:
        query = result.query.text

        reserved_for_output = (
            64  # might need to change depending on what the actual output look like
        )
        query_tokens = self.get_num_tokens(f"Query: {query} Document:  Relevant: ")

        few_shot_section = self._inference_handler._generate_fewshot_prompt(
            num_examples=self._num_few_shot_examples, examples=self._examples
        )
        few_shot_tokens = self.get_num_tokens(few_shot_section)

        max_doc_tokens = (
            self._context_size - few_shot_tokens - query_tokens - reserved_for_output
        )

        prompt = self._inference_handler.generate_prompt(
            result=result,
            index=index,
            max_doc_tokens=max_doc_tokens,
            tokenizer=self._tokenizer,
            num_few_shot_examples=self._num_few_shot_examples,
            fewshot_exampels=self._examples,
        )

        final_token_count = self.get_num_tokens(prompt)
        assert (
            final_token_count <= self._context_size - reserved_for_output
        ), f"Prompt overflow: {final_token_count} > {self._context_size - reserved_for_output}"

        return prompt, final_token_count

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def num_output_tokens(self) -> int:
        return 1

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

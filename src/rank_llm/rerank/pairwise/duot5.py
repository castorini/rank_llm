import logging
import math
from typing import List, Optional, Tuple

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.generation import GenerationConfig

from rank_llm.data import Result
from rank_llm.rerank.pairwise.pairwise_rankllm import PairwiseRankLLM

logger = logging.getLogger(__name__)


class DuoT5(PairwiseRankLLM):
    def __init__(
        self,
        model: str,
        prompt_mode: str = "duot5",
        context_size: int = 512,
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
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

        if num_few_shot_examples > 0:
            if not few_shot_file:
                raise ValueError(
                    "few_shot_examples_file must be provided when num_few_shot_examples > 0"
                )
            self._load_few_shot_examples(few_shot_file)

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

        tokenized = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self._context_size,
            return_tensors="pt",
        ).to(self._device)
        input_ids = tokenized["input_ids"]

        outputs = self._llm.generate(input_ids, generation_config=gen_cfg)
        output_ids = outputs.sequences
        logits = outputs.scores

        batch_outputs = [
            self._tokenizer.decode(
                seq,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            for seq in output_ids
        ]

        all_scores, all_output_token_counts = [], []
        # Use the logits from the generated token (logits[0] is of shape (batch_size, vocab_size))
        for logit_tensor in logits[0]:
            truth_logit = logit_tensor[self._true_id].item()
            false_logit = logit_tensor[self._false_id].item()
            score = math.exp(truth_logit) / (
                math.exp(truth_logit) + math.exp(false_logit)
            )
            all_scores.append(score)
            all_output_token_counts.append(self.num_output_tokens())

        return batch_outputs, all_output_token_counts, all_scores

    def run_llm(self, prompt: str) -> Tuple[str, int, float]:
        ret = self.run_llm_batched([prompt])
        return ret[0][0], ret[1][0], ret[2][0]

    def create_prompt(
        self, result: Result, index1: int, index2: int
    ) -> Tuple[str, int]:
        query = self._replace_number(result.query.text)

        reserved_for_output = (
            64  # might need to change depending on what the actual output look like
        )
        query_tokens = self.get_num_tokens(
            f"Query: {query} Document0:  Document1:  Relevant: "
        )

        few_shot_prompt = ""
        few_shot_tokens = 0
        few_shot_prompt = self._build_pairwise_few_shot_examples()
        few_shot_tokens = self.get_num_tokens(few_shot_prompt)

        max_token = (
            self._context_size - reserved_for_output - query_tokens - few_shot_tokens
        )

        doc1_raw = self.convert_doc_to_prompt_content(
            result.candidates[index1].doc, max_length=max_token
        )
        doc2_raw = self.convert_doc_to_prompt_content(
            result.candidates[index2].doc, max_length=max_token
        )

        doc1_tokens = self._tokenizer.encode(
            doc1_raw, truncation=True, max_length=max_token
        )
        doc2_tokens = self._tokenizer.encode(
            doc2_raw, truncation=True, max_length=max_token
        )

        doc1 = self._tokenizer.decode(doc1_tokens, skip_special_tokens=True)
        doc2 = self._tokenizer.decode(doc2_tokens, skip_special_tokens=True)

        prompt = f"{few_shot_prompt} Query: {query} Document0: {doc1} Document1: {doc2} Relevant: "
        prompt = prompt.replace("<unk>", "")

        return prompt, self.get_num_tokens(prompt)

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def num_output_tokens(self) -> int:
        return 1

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

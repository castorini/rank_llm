import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cmp_to_key
from typing import Dict, List, Optional, Tuple, Any

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.generation import GenerationConfig

from rank_llm.data import Candidate, Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM

try:
    from vllm import SamplingParams
except:
    LLM = None
    SamplingParams = None
logger = logging.getLogger(__name__)


class MonoT5(PointwiseRankLLM):
    def __init__(
        self,
        model: str,
        prompt_mode: str = "monot5",
        context_size: int = 512,
        device: str = "cuda"
    ):
        super().__init__(model=model, context_size=context_size, prompt_mode=prompt_mode, device=device)
        self._tokenizer = T5Tokenizer.from_pretrained(model)
        self._llm = T5ForConditionalGeneration.from_pretrained(
            model
        ).to(self._device)
        self._context_size = context_size

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

        token_prompts = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt' 
        ).to(self._device)

        token_prompts = token_prompts['input_ids']

        outputs = self._llm.generate(token_prompts, generation_config=gen_cfg)

        output_ids = outputs.sequences
        logits = outputs.scores

        outputs = [ 
            self._tokenizer.decode(
                single_token_sequence, skip_special_tokens=True, spaces_between_special_tokens=False
            ) 
            for single_token_sequence in output_ids
        ]

        output_token_counts = []
        scores = []

        for logit_tensor in logits[0]:
            truth_logit = logit_tensor[1176]
            false_logit = logit_tensor[6136]
            score = math.exp(truth_logit) / (math.exp(truth_logit) + math.exp(false_logit))
            scores.append(score)
            output_token_counts.append(self.num_output_tokens)

        return outputs, output_token_counts, scores
        
    def run_llm(
        self, prompt: str
    ) -> Tuple[str, int, float]:
        
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens()
        gen_cfg.min_new_tokens = self.num_output_tokens()
        gen_cfg.output_scores = True
        gen_cfg.return_dict_in_generate = True
        gen_cfg.do_sample = False

        token_prompt = self._tokenizer.encode(prompt, return_tensors="pt").to(
            self._device
        )
        output = self._llm.generate(token_prompt, generation_config=gen_cfg)
        output_ids = output.sequences
        logits = output.scores

        if self._llm.config.is_encoder_decoder:
            output_ids = output_ids[0]
            output_ids = output_ids[1:]

        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        truth_logit = logits[0][0][1176]
        false_logit = logits[0][0][6136]
        score = math.exp(truth_logit) / (math.exp(truth_logit) + math.exp(false_logit))

        return outputs, output_ids.size(0), score

    def num_output_tokens(self) -> int:
        return 1

    def create_prompt(
        self, result: Result, index: int
    ) -> Tuple[str, int]:
        query = result.query.text
        input = (
            f"Query: {query} Document: {result.candidates[index].doc['contents']}"
        )
        prompt = (
            self._tokenizer.decode(self._tokenizer.encode(input)[:(self._context_size - 32)])[:-4]
            + " Relevant: "
        )
        prompt = prompt.replace("<unk>", "")

        return prompt, self.get_num_tokens(prompt)

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

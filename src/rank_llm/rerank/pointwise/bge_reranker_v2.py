import logging
import math
from typing import List, Tuple
import torch

from FlagEmbedding import LayerWiseFlagLLMReranker, FlagLLMReranker, FlagReranker

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM

logger = logging.getLogger(__name__)


class BGE_RERANKER_V2(PointwiseRankLLM):
    def __init__(
        self,
        model: str,
        prompt_mode: str = "bge-reranker-v2",
        context_size: int = 8192,
        device: str = "cuda",
        batch_size: int = 32,
        use_fp16: bool = False
    ):
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            device=device,
            batch_size=batch_size,
        )

        if "base" in self._model or "large" in self._model or "m3" in self._model:
            self._llm = FlagReranker(self._model, use_fp16=use_fp16)
        elif "minicpm-layerwise" in self._model:
            self._llm=LayerWiseFlagLLMReranker(self._model, use_fp16=use_fp16)
        elif "gemma" in self._model:
            self._llm=FlagLLMReranker(self._model, use_fp16=use_fp16)
        else:
            raise ValueError("Given bge model doesn't exist or isn't supported in rank_llm.")

    def run_llm_batched(
        self,
        prompts: List[List[str]],
    ) -> Tuple[None, None, List[float]]:
        
        all_outputs = None
        all_output_token_counts = None
        all_scores = []

        pairs = prompts

        with torch.no_grad():
            if "base" in self._model or "large" in self._model or "m3" in self._model:
                all_scores = self._llm.compute_score(pairs)

            elif "gemma" in self._model:
                all_scores = self._llm.compute_score(pairs)

            elif "minicpm-layerwise" in self._model:
                scores = self._llm.compute_score(pairs, cutoff_layers=[28])
                if not isinstance(scores[0], float):
                    for score in scores[0]:
                        all_scores.append(score)
                else:
                    all_scores = scores

        return all_outputs, all_output_token_counts, all_scores
                
    def run_llm(self, prompt: str) -> Tuple[None, None, float]:

        outputs = None
        output_ids = None
        
        pair=prompt

        with torch.no_grad():
            if "base" in self._model or "large" in self._model or "m3" in self._model:
                score = self._llm.compute_score(pair)

            elif "gemma" in self._model:
                score = self._llm.compute_score(pair)

            elif "minicpm-layerwise" in self._model:
                score = self._llm.compute_score(pair, cutoff_layers=[28])

        return outputs, output_ids, score

    def num_output_tokens(self) -> int:
        return 1

    def create_prompt(self, result: Result, index: int) -> Tuple[str, int]:
        query = result.query.text
        prompt = [query, self.convert_doc_to_prompt_content(result.candidates[index].doc, max_length=self._context_size)]

        return prompt, None

    def get_num_tokens(self, prompt: str) -> int:
        return 1

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

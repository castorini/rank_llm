import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cmp_to_key
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.generation import GenerationConfig

from rank_llm.data import Candidate, Request, Result
from rank_llm.rerank.pairwise.pairwise_rankllm import PairwiseRankLLM
from rank_llm.rerank.rankllm import PromptMode

logger = logging.getLogger(__name__)


class DuoT5(PairwiseRankLLM):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode,
    ):
        super.__init__(model, context_size, prompt_mode)
        self._tokenizer = T5Tokenizer.from_pretrained("castorini/duot5-base-msmarco")
        self._llm = T5ForConditionalGeneration.from_pretrained(
            "castorini/duot5-base-msmarco"
        ).to(self._device)

    # TODO
    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: logging.Any,
    ) -> List[Result]:
        return

    # TODO
    def run_llm_batched(
        self, prompts: List[str | List[torch.Dict[str, str]]], **kwargs
    ) -> List[Tuple[str | int]]:
        return

    def run_llm(
        self, prompt: str, current_window_size: Optional[int] = None
    ) -> Tuple[str, int, float]:
        # CHANGE THIS CODE
        if current_window_size is None:
            current_window_size = self._window_size
        inputs = self._tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items()}
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens()
        gen_cfg.min_new_tokens = self.num_output_tokens()
        gen_cfg.decoder_start_token_id = None
        gen_cfg.output_scores = True
        gen_cfg.return_dict_in_generate = True
        # gen_cfg.temperature = 0
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
        # print(outputs, output_ids.size(0))
        return outputs, output_ids.size(0), score

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        # query = result.query.text
        # query = self._replace_number(query)
        # input = f"Query: {query} Document: {result.candidates[rank_start].doc['contents']}"
        # prompt = self._tokenizer.decode(self._tokenizer.encode(input)[:480])[:-4] + " Relevant: "
        # prompt = prompt.replace("<unk>","")

        # CHANGE THIS CODE
        query = result.query.text
        query = self._replace_number(query)
        doc1 = result.candidates[rank_start].doc["contents"]
        doc2 = result.candidates[rank_end].doc["contents"]
        doc1 = self._tokenizer.decode(self._tokenizer.encode(doc1)[:240])[:-4]
        doc2 = self._tokenizer.decode(self._tokenizer.encode(doc2)[:240])[:-4]
        prompt = f"Query: {query} Document0: {doc1} Document1: {doc2} Relevant:"
        prompt = prompt.replace("<unk>", "")
        return prompt, self.get_num_tokens(prompt)

    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in tqdm(chunks(results, batch_size), desc="Processing batches"):
                futures = [
                    executor.submit(self.create_prompt, result, rank_start, rank_end)
                    for result in batch
                ]
                completed_prompts = [
                    future.result() for future in as_completed(futures)
                ]
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        return 1

    def candidate_comparator(self, x: Candidate, y: Candidate) -> int:
        if x.score < y.score:
            return -1
        elif x.score > y.score:
            return 1
        else:
            return 0

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        return f"Given the query: {query}, output its relevance to the {num} documents."

    def _add_post_prompt(self, query: str, num: int) -> str:
        return f"Given the query: {query}, output its relevance to the {num} documents."

    def _add_few_shot_examples(self, conv):
        return 1
        # unused for now

    def permutation_pipeline(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) -> Result:
        """
        Runs the permutation pipeline on the passed in result set within the passed in rank range.

        Args:
            result (Result): The result object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The processed result object after applying permutation.
        """
        # CHANGE THIS CODE
        # print(len(result.candidates))
        # for i in range (len(result.candidates)):
        #     prompt, num_tokens = self.create_prompt(result, i, rank_end)
        #     output, output_num_tokens, score = self.run_llm(prompt=prompt)
        #     (result.candidates[i]).score = score

        # result.candidates.sort(key=cmp_to_key(self.candidate_comparator))
        n = len(result.candidates)
        scores = [0 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                else:
                    prompt1, num_tokens1 = self.create_prompt(result, i, j)
                    prompt2, num_tokens2 = self.create_prompt(result, j, i)
                    _, _, pi_j = self.run_llm(prompt=prompt1)
                    _, _, pj_i = self.run_llm(prompt=prompt2)
                    scores[i] = scores[i] + pi_j + 1 - pj_i

        for i in range(n):
            (result.candidates[i]).score = scores[i]

        result.candidates.sort(key=cmp_to_key(self.candidate_comparator))

        return result

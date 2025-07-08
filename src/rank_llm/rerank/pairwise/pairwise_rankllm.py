import copy
import logging
from abc import ABC
from datetime import datetime
from functools import cmp_to_key
from typing import Any, List, Optional, Tuple

from tqdm import tqdm

from rank_llm.data import Candidate, Request, Result
from rank_llm.rerank.rankllm import PromptMode, RankLLM

logger = logging.getLogger(__name__)


class PairwiseRankLLM(RankLLM, ABC):
    """
    Abstract base class that all pairwise rerankers implement.
    """

    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: Optional[PromptMode] = None,
        prompt_template_path: Optional[str] = None,
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        device: str = "cuda",
        filename: str = "",
        batch_size: int = 32,
    ) -> None:
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
        )
        self._num_few_shot_examples = num_few_shot_examples
        self._few_shot_file = few_shot_file
        self._device = device
        self._filename = filename
        self._batch_size = batch_size

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        """
        Re-rank candidates in a pairwise fashion:
         1. Build a list of all pairwise comparisons.
         2. Process in batches: create prompts, run LLM, update scores.
         3. Sort candidates by final score in descending order.
        """

        rerank_results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                invocations_history=[],
            )
            for request in requests
        ]

        # zero-initialize candidate scores
        for result in rerank_results:
            for candidate in result.candidates:
                candidate.score = 0

        num_queries, num_pairs = len(rerank_results), 0
        self._enumerated_indices = [[] for _ in range(num_queries)]
        for query_idx, res in enumerate(rerank_results):
            num_candidates = len(res.candidates)
            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    self._enumerated_indices[query_idx].append([i, j])
            num_pairs += len(self._enumerated_indices[query_idx])

        with tqdm(
            total=num_pairs, desc="Progress through (q, d) pairs"
        ) as progress_bar:
            for query_idx, pair_list in enumerate(self._enumerated_indices):
                index = 0
                while index < len(pair_list):
                    prompts, token_counts = self.create_prompt_batched(
                        rerank_results, query_idx, index
                    )

                    outputs, output_tokens, scores = self.run_llm_batched(prompts)

                    for (i, j), score in zip(
                        pair_list[index : index + len(scores)], scores
                    ):
                        rerank_results[query_idx].candidates[i].score += score
                        rerank_results[query_idx].candidates[j].score += 1 - score

                    index += self._batch_size
                    progress_bar.update(len(scores))

        for result in rerank_results:
            result.candidates.sort(key=cmp_to_key(self.candidate_comparator))

        return rerank_results

    def create_prompt_batched(
        self, results: List[Result], query_idx: int, index: int
    ) -> Tuple[List[str], List[int]]:
        """
        Create a batch of prompts for the given query_idx, taking pairs of candidates
        from self._enumerated_indices[query_idx] in the range [index : index + batch_size].
        """
        prompts, token_counts = [], []

        pair_list = self._enumerated_indices[query_idx]
        end_index = min(index + self._batch_size, len(pair_list))

        # Build prompts for each pair in [index, end_index)
        for pair_idx in range(index, end_index):
            i, j = pair_list[pair_idx]
            prompt, tcount = self.create_prompt(
                result=results[query_idx], index1=i, index2=j
            )
            prompts.append(prompt)
            token_counts.append(tcount)

        return prompts, token_counts

    def get_output_filename(
        self,
        top_k_candidates: int,
        dataset_name: str,
        shuffle_candidates: bool,
        **kwargs: Any,
    ) -> str:
        if self._filename != "":
            return self._filename
        _modelname = self._model.split("/")[-1]
        if _modelname.startswith("checkpoint"):
            _modelname = self._model.split("/")[-2] + "_" + _modelname
        name = f"{_modelname}_{self._context_size}_{top_k_candidates}"
        if dataset_name:
            name = f"{name}_{dataset_name}"

        if shuffle_candidates:
            self._filename = f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
        else:
            self._filename = f"{name}_{datetime.isoformat(datetime.now())}"

        return (
            f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
            if shuffle_candidates
            else f"{name}_{datetime.isoformat(datetime.now())}"
        )

    def candidate_comparator(self, x: Candidate, y: Candidate) -> int:
        if x.score < y.score:
            return 1
        elif x.score > y.score:
            return -1
        else:
            return 0

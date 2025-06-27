import copy
import logging
from abc import ABC
from datetime import datetime
from functools import cmp_to_key
from typing import Any, List, Optional, Tuple

from tqdm import tqdm

from rank_llm.data import Candidate, InferenceInvocation, Request, Result
from rank_llm.rerank.rankllm import PromptMode, RankLLM

logger = logging.getLogger(__name__)


class PointwiseRankLLM(RankLLM, ABC):
    """
    Abstract base class that all pointwise rerankers implement.

    All children of PointwiseRankLLM must implement these functions:
        - run_llm_batched
        - run_llm
        - create_prompt
        - get_num_tokens
        - cost_per_1k_tokens
        - num_output_tokens

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
        populate_invocations_history: bool = kwargs.get(
            "populate_invocations_history", False
        )
        rerank_results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                invocations_history=[],
            )
            for request in requests
        ]

        total_candidates = sum(len(result.candidates) for result in rerank_results)

        with tqdm(
            total=total_candidates, desc="Progress through (q, d) pairs"
        ) as progress_bar:
            index = 0
            while index < total_candidates:
                prompts, token_counts = self.create_prompt_batched(
                    results=rerank_results, index=index
                )

                outputs, output_token_counts, scores = self.run_llm_batched(
                    prompts=prompts
                )

                for i, score in enumerate(scores):
                    query_number, candidate_number = self.get_query_and_candidate_index(
                        rerank_results, index + i
                    )
                    rerank_results[query_number].candidates[
                        candidate_number
                    ].score = score
                    if populate_invocations_history:
                        inference_invocation = InferenceInvocation(
                            prompts[i],
                            outputs[i],
                            token_counts[i],
                            output_token_counts[i],
                        )
                        rerank_results[query_number].invocations_history.append(
                            inference_invocation
                        )

                progress_bar.update(len(scores))
                index += self._batch_size

        for result in rerank_results:
            result.candidates.sort(
                key=cmp_to_key(self.candidate_comparator), reverse=True
            )

        return rerank_results

    def get_query_and_candidate_index(
        self, results: List[Result], global_index: int
    ) -> Tuple[int, int]:
        cumulative_count = 0
        for query_index, result in enumerate(results):
            if global_index < cumulative_count + len(result.candidates):
                return query_index, global_index - cumulative_count
            cumulative_count += len(result.candidates)
        raise IndexError("Index out of range in get_query_and_candidate_index")

    def create_prompt_batched(
        self, results: List[Result], index: int
    ) -> Tuple[List[str], List[int]]:
        prompts = []
        token_counts = []

        for i in range(
            index,
            min(
                index + self._batch_size,
                sum(len(result.candidates) for result in results),
            ),
        ):
            query_number, candidate_number = self.get_query_and_candidate_index(
                results, i
            )
            prompt, token_count = self.create_prompt(
                result=results[query_number], index=candidate_number
            )

            prompts.append(prompt)
            token_counts.append(token_count)

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
        name = (
            f"{_modelname}_{self._context_size}_{top_k_candidates}_{self._prompt_mode}"
        )
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
            return -1
        elif x.score > y.score:
            return 1
        else:
            return 0

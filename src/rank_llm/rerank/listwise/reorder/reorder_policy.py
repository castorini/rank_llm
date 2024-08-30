import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TypeVar, Union

from rank_llm.data import Result

T = TypeVar("T")


@dataclass
class ModelFunction:
    # [(Result, SelectIndices)] -> [Prompt]
    create_prompt: Callable[
        [List[Tuple[Result, List[int]]]], List[Union[str, Dict[str, str]]]
    ]

    # [Prompt], [SelectedIndices] -> [Permutation]
    execute: Callable[
        [List[Union[str, Dict[str, str]]], List[List[int]]], List[List[int]]
    ]

    # Accepted Window Size
    window_size: int


class ReorderPolicy(ABC):
    @abstractmethod
    def reorder(
        self,
        requests: List[Result],
        rank_start: int,
        rank_end: int,
        model: ModelFunction,
        **kwargs,
    ) -> list[Result]:
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    def _shuffle_and_rescore(
        results: List[Result], select_indexes: List[int]
    ) -> List[Result]:
        # TODO: do nothing for now
        return results

    @staticmethod
    def _reorder_by_rank(items: List[T], idxes: List[int], rank: List[int]) -> List[T]:
        """
        Provide items, indexes, ranks, returns an ordered items, specifically ordered on idxes locations by rank
        """
        assert len(idxes) == len(rank)

        n = len(idxes)

        subset_item = [items[idxes[rank[i]]] for i in range(n)]

        for i in range(len(idxes)):
            items[idxes[i]] = subset_item[i]

        return items


class SlidingWindowReorderPolicy(ReorderPolicy):
    def __init__(
        self,
        step: int = 10,
        shuffle_candidates: bool = False,
        **kwargs,
    ):
        self._step_size = step

        self._shuffle_candidates = bool(shuffle_candidates)

    def reorder(
        self,
        requests: List[Result],
        rank_start: int,
        rank_end: int,
        model: ModelFunction,
        shuffle_candidates=False,
        logging=False,
        populate_exec_summary=False,
        **kwargs,
    ) -> List[Result]:
        window_size = model.window_size

        rerank_results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                ranking_exec_summary=[],
            )
            for request in requests
        ]

        if self._shuffle_candidates:
            self._shuffle_and_rescore(rerank_results, [*range(rank_start, rank_end)])

        # order of requests
        request_ranks = [[*range(len(request.candidates))] for request in requests]

        end_pos = rank_end
        start_pos = rank_end - window_size

        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + self._step_size != rank_start:
            # if logging:
            #     logger.info(f"start_pos: {start_pos}, end_pos: {end_pos}")
            start_pos = max(start_pos, rank_start)

            indices_working_on = [*range(start_pos, end_pos)]
            prompts = model.create_prompt(
                [
                    (request, [request_rank[i] for i in indices_working_on])
                    for request, request_rank in zip(requests, request_ranks)
                ]
            )
            orders = model.execute(prompts, [indices_working_on] * len(requests))

            for request_rank, order in zip(request_ranks, orders):
                self._reorder_by_rank(request_rank, indices_working_on, order)

            end_pos = end_pos - self._step_size
            start_pos = start_pos - self._step_size

        results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=self._reorder_by_rank(
                    copy.deepcopy(request.candidates),
                    [*range(len(request.candidates))],
                    rank,
                ),
                ranking_exec_summary=[],
            )
            for request, rank in zip(requests, request_ranks)
        ]

        for result, request in zip(results, requests):
            for j in range(len(result.candidates)):
                result.candidates[j].score = request.candidates[j].score

        return results

    @staticmethod
    def name() -> str:
        return "sliding_window"

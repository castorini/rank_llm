import copy
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TypeVar, Union

import numpy as np

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

    @abstractmethod
    def param_name(self):
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    def _shuffle_indices(indices: List[int]) -> List[int]:
        indices = list(indices)
        random.shuffle(indices)
        return indices

    @staticmethod
    def _shuffled(
        func: Callable[[List[Tuple[Result, List[int]]]], List[List[int]]]
    ) -> Callable[[List[Tuple[Result, List[int]]]], List[List[int]]]:
        def fun(batch: List[Tuple[Result, List[int]]]) -> List[List[int]]:
            perms = []
            perms_back = []
            batch_feed = []
            for res, ind in batch:
                perm = np.random.permutation(len(ind)).tolist()
                perm_back = [0 for _ in range(len(perm))]
                perms.append(perm)

                for i in range(len(perm)):
                    perm_back[perm[i]] = i

                batch_feed.append((res, [ind[x] for x in perm]))
                perms_back.append(perm_back)

            result_raw = func(batch)

            results = []
            for result, perm_back in zip(result_raw, perms_back):
                results.append([result[perm_back[x]] for x in range(len(result))])

            return results

        return fun

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
        step: int = None,
        extra_args: dict = None,
        **kwargs,
    ):
        self._step_size = (
            step
            if step is not None
            else extra_args.get("step_size", 10)
            if extra_args is not None
            else 10
        )
        self.coll = 0

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

        # order of requests
        if shuffle_candidates:
            request_ranks = [
                self._shuffle_indices(list(range(len(request.candidates))))
                for request in requests
            ]
        else:
            request_ranks = [
                list(range(len(request.candidates))) for request in requests
            ]

        end_pos = rank_end
        start_pos = rank_end - window_size

        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + self._step_size != rank_start:
            # if logging:
            #     logger.info(f"start_pos: {start_pos}, end_pos: {end_pos}")
            start_pos = max(start_pos, rank_start)

            indices_working_on = list(range(start_pos, end_pos))
            prompts = model.create_prompt(
                [
                    (request, [request_rank[i] for i in indices_working_on])
                    for request, request_rank in zip(requests, request_ranks)
                ]
            )
            orders = model.execute(
                prompts, [list(indices_working_on) for _ in requests]
            )

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

    def param_name(self):
        return f"slidingwindow_stp{self._step_size}"

    @staticmethod
    def name() -> str:
        return "sliding_window"

import copy
import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Literal, Tuple

from rank_llm.data import Result
from rank_llm.rerank.listwise.reorder.reorder_policy import ModelFunction, ReorderPolicy

logger = logging.getLogger(__name__)


@dataclass
class ReorderRequest:
    indices: List[int]
    result: List[int]


class TopDownReorderProcess:
    def __init__(
        self,
        top_k: int,
        window_size: int,
        pivot: int,
        indices: List[int],
        early_stop: int,
        padding: Literal["pad", "unpad"] = "pad",
    ):
        super().__init__()
        self._window_size = window_size
        self._pivot = pivot
        self._top_k = top_k
        self._indices = indices

        self._early_stop = early_stop

        self._padding = padding

    def _find_pivot(self, lst: List[int], piv: int) -> int:
        for i in range(len(lst)):
            if lst[i] == piv:
                return i
        # unreachable
        assert False

    def _fill_unchoose(self, lst: List[int]):
        st = set(lst)
        for x in self._indices:
            if x not in st:
                lst.append(x)

        return lst

    def _pad(self, lst: List[int]):
        if self._padding == "pad":
            st = set(lst)
            results = [x for x in lst]
            for i in reversed(self._indices):
                if len(results) >= self._window_size:
                    break
                if i not in st:
                    results.append(i)

            for i in reversed(self._indices):
                if len(results) >= self._window_size:
                    break
                results.append(i)

            return results
        else:
            return [x for x in lst]

    def _unpad(self, lst: List[int], result_perm: List[int]):
        if self._padding == "pad":
            return [x for x in result_perm if x < len(lst)]
        else:
            return result_perm

    def _remove_from_occ(self, lst: List[int], inds: List[int]):
        st = set(inds)
        return [x for x in lst if x not in st]

    def perform(self):
        top_k = self._top_k
        window_size = self._window_size
        pivot = self._pivot
        indices = [x for x in self._indices]

        assert pivot <= window_size
        assert top_k <= window_size

        """
        Algorithm is O(N^2) here, we can eliminate it to O(N) by split result into result and result_this_turn
        """

        while len(indices) > window_size:
            result = []

            # Notice this step will always only being run for 1 time, if pivot >= top_k
            #   which is also a parameter we want to control
            while len(result) < top_k:
                # base, find a pivot and do topdown
                base = indices[:window_size]
                request = ReorderRequest(self._pad(base), None)
                yield [request]
                base = [base[i] for i in self._unpad(base, request.result)]

                if len(base) < window_size:
                    for i in base:
                        if len(result) >= top_k:
                            break
                        result.append(i)
                    break

                piv_item = base[pivot]
                for i in range(pivot - 1):
                    result.append(base[i])

                if self._early_stop != -1 and self._early_stop < len(indices):
                    # early stop, then it won't directly feed all elements parallel

                    for i in range(window_size, len(indices), window_size - 1):
                        request_indices = [piv_item] + indices[i : i + window_size - 1]
                        request = ReorderRequest(self._pad(request_indices), None)
                        yield [request]
                        request_indices = [
                            request_indices[j]
                            for j in self._unpad(request_indices, request.result)
                        ]
                        loc = self._find_pivot(request_indices, piv_item)
                        result.extend(request_indices[:loc])

                        if len(result) >= self._early_stop:
                            break

                else:
                    # no early stop, parallelism
                    requests = []
                    req_inds = []

                    # then sort others
                    for i in range(window_size, len(indices), window_size - 1):
                        request_indices = [piv_item] + indices[i : i + window_size - 1]
                        req_inds.append(request_indices)
                        request = ReorderRequest(self._pad(request_indices), None)
                        requests.append(request)

                    yield requests

                    for request, request_indices, i in zip(
                        requests,
                        req_inds,
                        range(window_size, len(indices), window_size - 1),
                    ):
                        request_indices = [
                            request_indices[i]
                            for i in self._unpad(request_indices, request.result)
                        ]

                        # reordered
                        loc = self._find_pivot(request_indices, piv_item)
                        result.extend(request_indices[:loc])

                if len(result) + 1 == top_k:
                    result.append(piv_item)

                indices = self._remove_from_occ(indices, result)

            indices = result

        # finally, resort the value
        # here len(indices) == top_k
        request_indices = indices
        request = ReorderRequest(self._pad(request_indices), None)
        yield [request]
        indices = [
            request_indices[i] for i in self._unpad(request_indices, request.result)
        ]

        return self._fill_unchoose(indices)


def multiple_sort(
    requests: List[Result],
    indices_batch: List[List[int]],
    runner: Callable[[List[Tuple[Result, List[int]]]], List[List[int]]],
    window_size: int,
    pivot: int,
    top_k: int,
    early_stop: int,
) -> List[List[int]]:
    batch_size = len(requests)
    top_down_sorters = [
        TopDownReorderProcess(top_k, window_size, pivot, indices, early_stop=early_stop)
        for indices in indices_batch
    ]
    progress = [top_down_sorter.perform() for top_down_sorter in top_down_sorters]
    result: List[List[int]] = [None for _ in range(batch_size)]
    left_not_sorted = set(range(batch_size))

    while len(left_not_sorted) > 0:
        perm_request = []

        finish_requests = []
        for idx in left_not_sorted:
            try:
                reqs = next(progress[idx])
                perm_request.extend([(idx, req) for req in reqs])
            except StopIteration as e:
                result[idx] = e.value
                finish_requests.append(idx)
        for idx in finish_requests:
            left_not_sorted.remove(idx)

        outputs = runner([(requests[idx], req.indices) for idx, req in perm_request])

        for (idx, req), output in zip(perm_request, outputs):
            req.result = output

    return result


class TopDownReorderPolicy(ReorderPolicy):
    def __init__(
        self, top_k: int = 10, pivot: int = -1, early_stop: int = -1, **kwargs
    ):
        super().__init__()
        self._top_k = top_k
        self._pivot = pivot
        self._early_stop = early_stop

    def reorder(
        self,
        requests: List[Result],
        rank_start: int,
        rank_end: int,
        model: ModelFunction,
        shuffle_candidates: bool = False,
        **kwargs,
    ) -> list[Result]:
        window_size = model.window_size
        pivot = window_size // 2 if self._pivot < 0 else self._pivot

        runner: Callable[
            [List[Tuple[Result, List[int]]]], List[List[int]]
        ] = lambda reqs: model.execute(
            model.create_prompt(reqs), [ind for req, ind in reqs]
        )

        if shuffle_candidates:
            indices = [
                self._shuffle_indices(list(range(len(request.candidates))))
                for request in requests
            ]
        else:
            indices = [list(range(rank_start, rank_end)) for _ in range(len(requests))]

        request_ranks = multiple_sort(
            requests,
            indices,
            runner=runner,
            top_k=self._top_k,
            pivot=pivot,
            window_size=window_size,
            early_stop=self._early_stop,
        )

        results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=self._reorder_by_rank(
                    copy.deepcopy(request.candidates),
                    list(range(len(request.candidates))),
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
        return f"topdown_tpk{self._top_k}_pvt{self._pivot}"

    @staticmethod
    def name() -> str:
        return "top_down"

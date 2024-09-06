import copy
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

from rank_llm.data import Result

from .reorder_policy import ModelFunction, ReorderPolicy


@dataclass
class ResortRequest:
    indices: List[int]
    result: List[int]


class TournamentSortNode:
    @staticmethod
    def build(
        inds: List[int], window_size: int, top_k: int
    ) -> Tuple[
        "TournamentSortNode",
        List["TournamentSortNode"],
        Dict[int, "TournamentSortNode"],
    ]:
        assert window_size % top_k == 0

        cs: List["TournamentSortNode"] = [
            TournamentSortNode(top_k=top_k, index=x) for x in inds
        ]

        base_nodes = {idx: c for idx, c in zip(inds, cs)}
        all_cs: List["TournamentSortNode"] = []
        all_cs.extend(cs)

        dq = deque(all_cs)

        while len(dq) > 1:
            cnt = 0
            children = []

            while (len(dq) != 0) and (cnt + dq[0].estimate_size() <= window_size):
                children.append(dq[0])
                cnt += dq[0].estimate_size()
                dq.popleft()

            if len(children) == 1:
                child = children[0]
                dq.append(child)
            else:
                nd = TournamentSortNode(top_k=top_k, children=children)
                all_cs.append(nd)
                dq.append(nd)

        return dq[0], all_cs, base_nodes

    def __init__(
        self,
        top_k: int,
        *,
        children: Union[List["TournamentSortNode"]] = None,
        index: int = None,
    ):
        super().__init__()

        self.parent: "TournamentSortNode" = None

        self._top_k = top_k

        if children is not None:
            for child in children:
                child.parent = self

            self._n = len(children)
            self._children = children
            self._top: List[int] = None
            self._tmp: List[int] = None
        else:
            self._n = -1
            self._index = index
            self._top: List[int] = [index]
            self._tmp: List[int] = None

    def reset(self):
        if self._n == -1:
            return
        self._top = None

    def invalidate(self):
        if self._n != -1:
            return

        self._top = []

    def get_resort_param(self) -> Union[List[int], None]:
        if self._n == -1 or self._top is not None:
            return None
        self._tmp = [x for child in self._children for x in child.top()]
        return [ind for ind in self._tmp]

    def resort(self, perm: List[int]):
        assert self._tmp is not None and self._top is None

        tops = []
        for i in perm:
            if len(tops) > self._top_k:
                break
            ind = self._tmp[i]
            if ind not in tops:
                tops.append(ind)

        self._top = tops

        return

    def top(self) -> List[int]:
        assert self._top is not None
        return self._top[: min(len(self._top), self._top_k)]

    def estimate_size(self) -> int:
        if self._n == -1:
            return 1
        else:
            return self._top_k

    def __str__(self):
        if self._n == -1:
            return f"[{self._index}]"
        else:
            return f"({' '.join([str(x) for x in self._children])})"


class TournamentSorter:
    def _get_random_indices(
        self, expect_size: int, ind_choices: List[int]
    ) -> List[int]:
        choices = set(ind_choices)
        result = []
        for j in reversed(range(self._n_passage)):
            if len(result) + len(ind_choices) >= expect_size:
                break
            if j not in choices:
                result.append(j)

        for j in reversed(range(self._n_passage)):
            if len(result) + len(ind_choices) >= expect_size:
                break
            result.append(j)
        return result

    def _pad_size(self, inds: List[int]) -> List[int]:
        if len(inds) >= self._window_size:
            return inds
        else:
            fitters = self._get_random_indices(self._window_size, inds)
            return inds + fitters

    def _unpad_perm(self, inds: List[int], padded: List[int], perm: List[int]):
        return [x for x in perm if x < len(inds)]

    def _fill_up(self, result: List[int]) -> List[int]:
        result_set = set(result)
        filled_up_result = [x for x in result]
        for idx in self._indices:
            if idx not in result_set:
                filled_up_result.append(idx)
        return filled_up_result

    def __init__(self, indices: List[int], window_size: int, r: int):
        super().__init__()
        self._window_size = window_size
        self._r = r

        self._n_passage = len(indices)

        self._indices = indices

        self._tr, self._all_node, self._idx_to_node = TournamentSortNode.build(
            indices, window_size=window_size, top_k=r
        )

        self.count_inference = 0

    def _pop(self, x: int) -> List[TournamentSortNode]:
        on: TournamentSortNode = self._idx_to_node[x]
        lst = []
        while on is not None:
            lst.append(on)
            on.invalidate()
            on.reset()
            on = on.parent
        return lst

    def perform(self, top_k: int):
        result = []

        # firstly, simple sort
        for nd in self._all_node:
            resort_param = nd.get_resort_param()
            if resort_param is not None:
                padded = self._pad_size(resort_param)
                request = ResortRequest(padded, [])
                yield request
                self.count_inference += 1
                cleaned_result = self._unpad_perm(resort_param, padded, request.result)
                nd.resort(cleaned_result)

        while len(result) < top_k:
            tpv = self._tr.top()[0]
            result.append(tpv)
            nodes = self._pop(tpv)

            if len(result) >= top_k:
                break

            for node in nodes:
                resort_param = node.get_resort_param()
                if resort_param is not None:
                    padded = self._pad_size(resort_param)
                    request = ResortRequest(padded, [])
                    yield request
                    self.count_inference += 1
                    assert len(request.result) > 0
                    cleaned_result = self._unpad_perm(
                        resort_param, padded, request.result
                    )
                    node.resort(cleaned_result)

        return self._fill_up(result)


def multiple_sort(
    requests: List[Result],
    indices_batch: List[List[int]],
    runner: Callable[[List[Tuple[Result, List[int]]]], List[List[int]]],
    window_size: int,
    r: int,
    top_k: int,
):
    batch_size = len(requests)
    tournament_sorters: List[TournamentSorter] = [
        TournamentSorter(indices, window_size, r) for indices in indices_batch
    ]
    progress = [
        tournament_sorter.perform(top_k) for tournament_sorter in tournament_sorters
    ]
    result = [None for _ in range(batch_size)]
    left_not_sorted = set(range(batch_size))

    while len(left_not_sorted) > 0:
        perm_request = []

        finish_requests = []
        for idx in left_not_sorted:
            try:
                req = next(progress[idx])
                perm_request.append((idx, req))
            except StopIteration as e:
                result[idx] = e.value
                finish_requests.append(idx)
        for idx in finish_requests:
            left_not_sorted.remove(idx)

        outputs = runner([(requests[idx], req.indices) for idx, req in perm_request])

        for (idx, req), output in zip(perm_request, outputs):
            req.result = output

    return result


class TournamentSortReorderPolicy(ReorderPolicy):
    def __init__(self, top_k: int = 10, r: int = 1, **kwargs):
        super().__init__()
        self._top_k = top_k
        self._r = r

    def reorder(
        self,
        requests: List[Result],
        rank_start: int,
        rank_end: int,
        model: ModelFunction,
        **kwargs,
    ) -> list[Result]:
        window_size = model.window_size

        runner: Callable[
            [List[Tuple[Result, List[int]]]], List[List[int]]
        ] = lambda reqs: model.execute(
            model.create_prompt(reqs), [ind for req, ind in reqs]
        )

        request_ranks = multiple_sort(
            requests,
            [list(range(rank_start, rank_end)) for _ in range(len(requests))],
            runner=runner,
            window_size=window_size,
            top_k=self._top_k,
            r=self._r,
        )

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
        return "tournament_sort"

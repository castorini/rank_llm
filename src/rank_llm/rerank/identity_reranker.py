import copy
import random
from typing import List

from rank_llm.data import Request, Result


class IdentityReranker:
    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
    ) -> List[Result]:
        """
        A trivial reranker that returns a subsection of the retrieved candidates list as-is or shuffled.

        Args:
            requests (List[Request]): The list of requests. Each request has a query and a candidates list.
            rank_start (int, optional): The starting rank for returning. Defaults to 0.
            rank_end (int, optional): The end rank for returning. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before returning. Defaults to False.

        Returns:
            List[Result]: A list containing the reranked candidates.
        """
        results = []
        for request in requests:
            rerank_result = Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                ranking_exec_summary=[],
            )
            if shuffle_candidates:
                # Randomly shuffle rerank_result between rank_start and rank_end
                rerank_result.candidates[rank_start:rank_end] = random.sample(
                    rerank_result.candidates[rank_start:rank_end],
                    len(rerank_result.candidates[rank_start:rank_end]),
                )
            print(f"rerank result: {rerank_result}")
            results.append(rerank_result)
        return results

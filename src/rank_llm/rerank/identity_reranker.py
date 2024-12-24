import copy
import random
from datetime import datetime
from typing import Any, List

from rank_llm.data import Request, Result


class IdentityReranker:
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
            results.append(rerank_result)
        return results

    def get_name(self) -> str:
        return "identity_reranker"

    def get_output_filename(
        self,
        top_k_candidates: int,
        dataset_name: str,
        shuffle_candidates: bool,
        **kwargs: Any,
    ) -> str:
        return f"identity_{datetime.isoformat(datetime.now())}"

from importlib.resources import files
from typing import Any, List, Optional

import torch

from rank_llm.data import Request, Result
from rank_llm.rerank.listwise.rank_fid import RankFiDDistill, RankFiDScore
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.reranker import Reranker

TEMPLATES = files("rank_llm.rerank.prompt_templates")


class LiT5DistillReranker:
    def __init__(
        self,
        model_path: str = "castorini/LiT5-Distill-base",
        context_size: int = 300,
        prompt_mode: Optional[PromptMode] = None,
        prompt_template_path: str = (TEMPLATES / "rank_fid_template.yaml"),
        precision: str = "bfloat16",
        window_size: int = 20,
        stride: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ) -> None:
        model_coordinator = RankFiDDistill(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            precision=precision,
            window_size=window_size,
            stride=stride,
            device=device,
            batch_size=batch_size,
        )
        self._reranker = Reranker(model_coordinator)

    def rerank_batch(
        self,
        requests: list[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        """
        Reranks a list of requests using the LiT5-Distill model.

        Args:
            requests (List[Request]): The list of requests. Each request has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window size.
        Returns:
            List[Result]: A list containing the reranked results.

        Note:
            check 'reranker.rerank' for implementation details of reranking process.
        """
        return self._reranker.rerank_batch(
            requests=requests,
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            **kwargs,
        )

    def rerank(
        self,
        request: Request,
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> Result:
        """
        Reranks a request using the LiT5-Distill model.

        Args:
            request (Request): The reranking request which has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window size.
        Returns:
            Result: the rerank result which contains the reranked candidates.

        Note:
            check 'reranker.rerank' for implementation details of reranking process.
        """
        return self._reranker.rerank_batch(
            requests=[request],
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            **kwargs,
        )


class LiT5ScoreReranker:
    def __init__(
        self,
        model_path: str = "castorini/LiT5-Score-base",
        context_size: int = 300,
        prompt_mode: Optional[PromptMode] = None,
        prompt_template_path: str = (TEMPLATES / "rank_fid_score_template.yaml"),
        window_size: int = 20,
        stride: int = 10,
        runfile_path: str = "runs/run.${topics}_${firststage}_${model//\//}",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ) -> None:
        model_coordinator = RankFiDScore(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            window_size=window_size,
            stride=stride,
            device=device,
            batch_size=batch_size,
        )
        self._reranker = Reranker(model_coordinator)

    def rerank_batch(
        self,
        requests: list[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        """
        Reranks a list of requests using the LiT5-Score model.

        Args:
            requests (List[Request]): The list of requests. Each request has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window size.
        Returns:
            List[Result]: A list containing the reranked results.

        Note:
            check 'reranker.rerank' for implementation details of reranking process.
        """
        return self._reranker.rerank_batch(
            requests=requests,
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            **kwargs,
        )

    def rerank(
        self,
        request: Request,
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> Result:
        """
        Reranks a request using the LiT5-Score model.

        Args:
            request (Request): The reranking request which has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window size.
        Returns:
            Result: the rerank result which contains the reranked candidates.

        Note:
            check 'reranker.rerank' for implementation details of reranking process.
        """
        return self._reranker.rerank_batch(
            requests=[request],
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            **kwargs,
        )

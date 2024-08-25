import logging
from abc import ABC

from rank_llm.rerank.rankllm import RankLLM

logger = logging.getLogger(__name__)


class PairwiseRankLLM(RankLLM, ABC):
    def __init__(
        self,
        model: str,
        device: str = "cuda",
        window_size: int = 20,
        batched: bool = False,
    ) -> None:
        super.__init__(model)
        self._window_size = window_size
        self._device = device
        self._batched = batched

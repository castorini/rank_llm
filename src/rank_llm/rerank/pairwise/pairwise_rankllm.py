import logging
from abc import ABC

from rank_llm.rerank.rankllm import PromptMode, RankLLM

try:
    from vllm import LLM, SamplingParams
except:
    LLM = None
    SamplingParams = None

logger = logging.getLogger(__name__)


class PairwiseRankLLM(RankLLM, ABC):
    """
    All children of PairwiseRankLLM must implement these functions:
        - currently all abstract functions of RankLLM
    """

    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode,
        device: str = "cuda",
        window_size: int = 20,
        batched: bool = False,
    ) -> None:
        super.__init__(model, context_size, prompt_mode)
        self._device = device
        self._window_size = window_size
        self._batched = batched

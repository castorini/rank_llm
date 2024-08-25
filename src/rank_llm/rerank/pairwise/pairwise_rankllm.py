import logging
from abc import ABC

from rank_llm.rerank.rankllm import PromptMode, RankLLM

logger = logging.getLogger(__name__)


class PairwiseRankLLM(RankLLM, ABC):
    """
    Abstract base class that all pairwise rerankers implement.

    All concrete children of RankLLM must implement these functions:
        - rerank_batch
        - run_llm_batched
        - run_llm
        - create_prompt_batched
        - create_prompt
        - get_num_tokens
        - cost_per_1k_tokens
        - num_output_tokens
    """

    def __init__(self, model: str, context_size: int, prompt_mode: PromptMode) -> None:
        super.__init__(model, context_size, prompt_mode)

    # TODO
    def get_output_filename(
        self,
        top_k_candidates: int,
        dataset_name: str,
        shuffle_candidates: bool,
        **kwargs: logging.Any,
    ) -> str:
        return

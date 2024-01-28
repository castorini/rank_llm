from typing import List

from rank_llm.rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.reranker import Reranker
from rank_llm.result import Result


class ZephyrReranker:
    def __init__(
        self,
        model_path: str = "castorini/rank_zephyr_7b_v1_full",
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = True,
        window_size: int = 20,
        system_message: str = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query",
    ):
        agent = RankListwiseOSLLM(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
            system_message=system_message,
        )
        self._reranker = Reranker(agent)

    def rerank(
        self,
        retrieved_results: List[Result],
        rank_start: int = 0,
        rank_end: int = 100,
        window_size: int = 20,
        step: int = 10,
        shuffle_candidates: bool = False,
        logging: bool = False,
    ):
        return self._reranker.rerank(
            retrieved_results=retrieved_results,
            rank_start=rank_start,
            rank_end=rank_end,
            window_size=window_size,
            step=step,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
        )

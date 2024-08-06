import logging
from abc import ABC
from tqdm import tqdm
import copy
from functools import cmp_to_key
from datetime import datetime
from rank_llm.rerank.rankllm import PromptMode, RankLLM
from typing import List, Any, Tuple
from rank_llm.data import Result, Request, Candidate

try:
    from vllm import LLM, SamplingParams
except:
    LLM = None
    SamplingParams = None

logger = logging.getLogger(__name__)


class PointwiseRankLLM(RankLLM, ABC):
    """
    All children of PointwiseRankLLM must implement these functions:
        - currently all abstract functions of RankLLM

    """

    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode,
        device: str = "cuda",
        filename: str = ""
    ) -> None:
        super().__init__(model, context_size, prompt_mode)
        self._device = device
        self._filename = filename

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        rerank_results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                ranking_exec_summary=[]
            )
            for request in requests
        ]
        for index in tqdm(range(len(rerank_results[0].candidates))):
            prompts, token_counts = self.create_prompt_batched(
                results=rerank_results, index=index
            )
            outputs, output_tokens, scores = self.run_llm_batched(prompts=prompts)
            for result, score in zip(rerank_results, scores):
                result.candidates[index].score = score
        
        for result in rerank_results:
            result.candidates.sort(key=cmp_to_key(self.candidate_comparator))

        return rerank_results

    def create_prompt_batched(
        self,
        results: List[Result],
        index
    ) -> Tuple[List[str], List[int]]:
        prompts = []
        token_counts = []

        for result in results:
            prompt, token_count = self.create_prompt(result=result, index=index)
            prompts.append(prompt)
            token_counts.append(token_count)

        return prompts, token_counts


    def candidate_comparator(self, x: Candidate, y: Candidate) -> int:
        if x.score < y.score:
            return -1
        elif x.score > y.score:
            return 1
        else:
            return 0

    def get_output_filename(
        self,
        top_k_candidates: int,
        dataset_name: str,
        shuffle_candidates: bool,
        **kwargs: Any,
    ) -> str:
        if (self._filename != ""):
            return self._filename
        _modelname = self._model.split("/")[-1]
        if _modelname.startswith("checkpoint"):
            _modelname = self._model.split("/")[-2] + "_" + _modelname
        name = (
            f"{_modelname}_{self._context_size}_{top_k_candidates}_{self._prompt_mode}"
        )
        if dataset_name:
            name = f"{name}_{dataset_name}"

        if shuffle_candidates:
            self._filename = f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
        else:
            self._filename = f"{name}_{datetime.isoformat(datetime.now())}"

        return (
            f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
            if shuffle_candidates
            else f"{name}_{datetime.isoformat(datetime.now())}"
        )
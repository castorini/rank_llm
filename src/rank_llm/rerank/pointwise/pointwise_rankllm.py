import copy
import logging
import math
import re
from abc import ABC
from datetime import datetime
from functools import cmp_to_key
from typing import Any, Dict, List, Tuple

from ftfy import fix_text
from tqdm import tqdm

from rank_llm.data import Candidate, Request, Result
from rank_llm.rerank.rankllm import PromptMode, RankLLM

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
        filename: str = "",
        batch_size: int = 32,
    ) -> None:
        super().__init__(model, context_size, prompt_mode)
        self._device = device
        self._filename = filename
        self._batch_size = batch_size

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
                ranking_exec_summary=[],
            )
            for request in requests
        ]
        end = len(rerank_results[0].candidates) * len(requests)

        with tqdm(total=end, desc="Progress through (q, d) pairs") as progress_bar:
            for index in range(0, end, self._batch_size):
                prompts, token_counts = self.create_prompt_batched(
                    results=rerank_results, index=index
                )

                outputs, output_tokens, scores = self.run_llm_batched(prompts=prompts)

                for update_index in range(
                    index,
                    min(
                        index + self._batch_size,
                        len(rerank_results[0].candidates) * len(rerank_results),
                    ),
                ):
                    query_number = math.floor(
                        update_index / len(rerank_results[0].candidates)
                    )
                    candidate_number = update_index % len(rerank_results[0].candidates)

                    rerank_results[query_number].candidates[
                        candidate_number
                    ].score = scores[update_index - index]

                if index + self._batch_size > end:
                    progress_bar.update(end - index)
                else:
                    progress_bar.update(self._batch_size)

        for result in rerank_results:
            result.candidates.sort(
                key=cmp_to_key(self.candidate_comparator), reverse=True
            )

        return rerank_results

    def create_prompt_batched(
        self, results: List[Result], index
    ) -> Tuple[List[str], List[int]]:
        prompts = []
        token_counts = []

        for index in range(
            index,
            min(index + self._batch_size, len(results[0].candidates) * len(results)),
        ):
            query_number = math.floor(index / len(results[0].candidates))
            candidate_number = index % len(results[0].candidates)

            prompt, token_count = self.create_prompt(
                result=results[query_number], index=candidate_number
            )

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
        if self._filename != "":
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

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def convert_doc_to_prompt_content(
        self, doc: Dict[str, Any], max_length: int
    ) -> str:
        if "text" in doc:
            content = doc["text"]
        elif "segment" in doc:
            content = doc["segment"]
        elif "contents" in doc:
            content = doc["contents"]
        elif "content" in doc:
            content = doc["content"]
        elif "body" in doc:
            content = doc["body"]
        else:
            content = doc["passage"]
        if "title" in doc and doc["title"]:
            content = "Title: " + doc["title"] + " " + "Content: " + content
        content = content.strip()
        content = fix_text(content)
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = " ".join(content.split()[: int(max_length)])
        return self._replace_number(content)

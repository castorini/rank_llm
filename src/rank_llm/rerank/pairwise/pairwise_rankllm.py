import copy
import json
import logging
import random
import re
from abc import ABC
from datetime import datetime
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Tuple

from ftfy import fix_text
from tqdm import tqdm

from rank_llm.data import Candidate, Request, Result
from rank_llm.rerank.rankllm import PromptMode, RankLLM

logger = logging.getLogger(__name__)


class PairwiseRankLLM(RankLLM, ABC):
    """
    Abstract base class that all pairwise rerankers implement.
    """

    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode,
        num_few_shot_examples: int,
        few_shot_file: Optional[str],
        device: str = "cuda",
        filename: str = "",
        batch_size: int = 32,
    ) -> None:
        super().__init__(model, context_size, prompt_mode)
        self._num_few_shot_examples = num_few_shot_examples
        self._few_shot_file = few_shot_file
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
        """
        Re-rank candidates in a pairwise fashion:
         1. Build a list of all pairwise comparisons.
         2. Process in batches: create prompts, run LLM, update scores.
         3. Sort candidates by final score in descending order.
        """

        rerank_results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                invocations_history=[],
            )
            for request in requests
        ]

        # zero-initialize candidate scores
        for result in rerank_results:
            for candidate in result.candidates:
                candidate.score = 0

        num_queries, num_pairs = len(rerank_results), 0
        self._enumerated_indices = [[] for _ in range(num_queries)]
        for query_idx, res in enumerate(rerank_results):
            num_candidates = len(res.candidates)
            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    self._enumerated_indices[query_idx].append([i, j])
            num_pairs += len(self._enumerated_indices[query_idx])

        with tqdm(
            total=num_pairs, desc="Progress through (q, d) pairs"
        ) as progress_bar:
            for query_idx, pair_list in enumerate(self._enumerated_indices):
                index = 0
                while index < len(pair_list):
                    prompts, token_counts = self.create_prompt_batched(
                        rerank_results, query_idx, index
                    )

                    outputs, output_tokens, scores = self.run_llm_batched(prompts)

                    for (i, j), score in zip(
                        pair_list[index : index + len(scores)], scores
                    ):
                        rerank_results[query_idx].candidates[i].score += score
                        rerank_results[query_idx].candidates[j].score += 1 - score

                    index += self._batch_size
                    progress_bar.update(len(scores))

        for result in rerank_results:
            result.candidates.sort(key=cmp_to_key(self.candidate_comparator))

        return rerank_results

    def create_prompt_batched(
        self, results: List[Result], query_idx: int, index: int
    ) -> Tuple[List[str], List[int]]:
        """
        Create a batch of prompts for the given query_idx, taking pairs of candidates
        from self._enumerated_indices[query_idx] in the range [index : index + batch_size].
        """
        prompts, token_counts = [], []

        pair_list = self._enumerated_indices[query_idx]
        end_index = min(index + self._batch_size, len(pair_list))

        # Build prompts for each pair in [index, end_index)
        for pair_idx in range(index, end_index):
            i, j = pair_list[pair_idx]
            prompt, tcount = self.create_prompt(
                result=results[query_idx], index1=i, index2=j
            )
            prompts.append(prompt)
            token_counts.append(tcount)

        return prompts, token_counts

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

    def candidate_comparator(self, x: Candidate, y: Candidate) -> int:
        if x.score < y.score:
            return 1
        elif x.score > y.score:
            return -1
        else:
            return 0

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

    def _load_few_shot_examples(self, file_path: str):
        try:
            with open(file_path, "r") as json_file:
                self._examples = json.load(json_file)
        except FileNotFoundError:
            raise ValueError(f"Few-shot examples file not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(
                f"Invalid JSON format in few-shot examples file: {file_path}"
            )

    def _build_pairwise_few_shot_examples(self) -> str:
        if self._num_few_shot_examples > 0 and hasattr(self, "_examples"):
            examples = []
            for _ in range(min(self._num_few_shot_examples, len(self._examples))):
                ex = random.choice(self._examples)
                try:
                    # assume each value for conversation contain 2 values (user query + docs, asssistant response)
                    example_query = (
                        ex["conversations"][0]["value"]
                        .split("Query: ")[-1]
                        .split("Document0: ")[0]
                        .strip()
                    )
                    example_doc0 = (
                        ex["conversations"][0]["value"]
                        .split(" Document0: ")[-1]
                        .split("Document1: ")[0]
                        .strip()
                    )
                    example_doc1 = (
                        ex["conversations"][0]["value"]
                        .split(" Document1: ")[-1]
                        .strip()
                    )
                    example_relevance = ex["conversations"][1]["value"].strip()

                    examples.append(
                        f"Query: {example_query} Document0: {example_doc0} Document1: {example_doc1} Relevant: {example_relevance}"
                    )
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

            return "\n\n".join(examples) + "\n\n" if examples else ""
        else:
            return ""

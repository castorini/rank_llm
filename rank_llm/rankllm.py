from abc import ABC, abstractmethod
import copy
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import random
from typing import List, Union, Dict, Any, Tuple

from tqdm import tqdm


class PromptMode(Enum):
    UNSPECIFIED = "unspecified"
    RANK_GPT = "rank_GPT"
    LRL = "LRL"

    def __str__(self):
        return self.value


class RankLLM(ABC):
    def __init__(
        self,
        model: str,
        context_size: int,
        top_k_candidates: int,
        dataset: str,
        prompt_mode: PromptMode,
    ) -> None:
        self._model = model
        self._context_size = context_size
        self._top_k_candidates = top_k_candidates
        self._dataset = dataset
        self._prompt_mode = prompt_mode

    def max_tokens(self) -> int:
        return self._context_size

    @abstractmethod
    def run_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> Tuple[str, int]:
        pass

    @abstractmethod
    def create_prompt(
        self, retrieved_result: Dict[str, Any], rank_start: int, rank_end: int
    ) -> Tuple[Union[str, List[Dict[str, str]]], int]:
        pass

    @abstractmethod
    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        pass

    @abstractmethod
    def cost_per_1k_token(self, input_token: bool) -> float:
        pass

    @abstractmethod
    def num_output_tokens(self) -> int:
        pass

    def permutation_pipeline(
        self,
        result: Dict[str, Any],
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ):
        prompt, in_token_count = self.create_prompt(result, rank_start, rank_end)
        if logging:
            print(f"prompt: {prompt}")
        permutation, out_token_count = self.run_llm(prompt)
        if logging:
            print(f"\noutput: {permutation}")
        rerank_result = self.receive_permutation(
            result, permutation, rank_start, rank_end
        )
        return rerank_result, in_token_count, out_token_count, prompt, permutation

    def sliding_windows(
        self,
        retrieved_result: Dict[str, Any],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
    ):
        in_token_count = 0
        out_token_count = 0
        rerank_result = copy.deepcopy(retrieved_result)
        if shuffle_candidates:
            # First randomly shuffle rerank_result between rank_start and rank_end
            rerank_result["hits"][rank_start:rank_end] = random.sample(
                rerank_result["hits"][rank_start:rank_end],
                len(rerank_result["hits"][rank_start:rank_end]),
            )
            # Next rescore all candidates with 1/rank
            for i, hit in enumerate(rerank_result["hits"]):
                hit["score"] = 1.0 / (i + 1)
                hit["rank"] = i + 1
        end_pos = rank_end
        start_pos = rank_end - window_size
        prompts = []
        permutations = []
        while start_pos >= rank_start:

            # print(f"\nrerank_results={rerank_result}")
            # print(f"start_pos={start_pos}")
            # print(f"rank_start={rank_start}")

            start_pos = max(start_pos, rank_start)
            (
                rerank_result,
                in_count,
                out_count,
                prompt,
                permutation,
            ) = self.permutation_pipeline(rerank_result, start_pos, end_pos, logging)
            in_token_count += in_count
            out_token_count += out_count
            prompts.append(prompt)
            permutations.append(permutation)
            end_pos = end_pos - step
            start_pos = start_pos - step
        return rerank_result, in_token_count, out_token_count, prompts, permutations

    def get_ranking_cost_upperbound(
        self, num_q: int, rank_start: int, rank_end: int, window_size: int, step: int
    ) -> Tuple[float, int]:
        # For every prompt generated for every query assume the max context size is used.
        num_promt = (rank_end - rank_start - window_size) / step + 1
        input_token_count = (
            num_q * num_promt * (self._context_size - self.num_output_tokens())
        )
        output_token_count = num_q * num_promt * self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def get_ranking_cost(
        self,
        retrieved_results: List[Dict[str, Any]],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
    ) -> Tuple[float, int]:
        input_token_count = 0
        output_token_count = 0
        # Go through the retrieval result using the sliding window and count the number of tokens for generated prompts.
        # This is an estimated cost analysis since the actual prompts' length will depend on the ranking.
        for result in tqdm(retrieved_results):
            end_pos = rank_end
            start_pos = rank_end - window_size
            while start_pos >= rank_start:
                start_pos = max(start_pos, rank_start)
                prompt, _ = self.create_prompt(result, start_pos, end_pos)
                input_token_count += self.get_num_tokens(prompt)
                end_pos = end_pos - step
                start_pos = start_pos - step
                output_token_count += self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def _clean_response(self, response: str) -> str:
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def receive_permutation(
        self, item: Dict[str, Any], permutation: str, rank_start: int, rank_end: int
    ) -> Dict[str, Any]:
        response = self._clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(item["hits"][rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        for j, x in enumerate(response):
            item["hits"][j + rank_start] = copy.deepcopy(cut_range[x])
            if "rank" in item["hits"][j + rank_start]:
                item["hits"][j + rank_start]["rank"] = cut_range[j]["rank"]
            if "score" in item["hits"][j + rank_start]:
                item["hits"][j + rank_start]["score"] = cut_range[j]["score"]
        return item

    def write_rerank_results(
        self,
        retrieval_method_name: str,
        rerank_results: List[Dict[str, Any]],
        input_token_counts: List[int],
        output_token_counts: List[int],
        # List[str] for Vicuna, List[List[Dict[str, str]]] for gpt models.
        prompts: Union[List[str], List[List[Dict[str, str]]]],
        responses: List[str],
        shuffle_candidates: bool = False,
    ) -> str:
        # write rerank results
        Path(f"../rerank_results/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        _modelname = self._model.split("/")[-1]
        name = f"{_modelname}_{self._context_size}_{self._top_k_candidates}_{self._prompt_mode}_{self._dataset}"
        name = (
            f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
            if shuffle_candidates
            else f"{name}_{datetime.isoformat(datetime.now())}"
        )
        result_file_name = f"../rerank_results/{retrieval_method_name}/{name}.txt"
        with open(result_file_name, "w") as f:
            for i in range(len(rerank_results)):
                rank = 1
                hits = rerank_results[i]["hits"]
                for hit in hits:
                    f.write(
                        f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n"
                    )
                    rank += 1
        # Write token counts
        Path(f"../token_counts/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        count_file_name = f"../token_counts/{retrieval_method_name}/{name}.txt"
        counts = {}
        for i, (in_count, out_count) in enumerate(
            zip(input_token_counts, output_token_counts)
        ):
            counts[rerank_results[i]["query"]] = (in_count, out_count)
        with open(count_file_name, "w") as f:
            json.dump(counts, f, indent=4)
        # Write prompts and responses
        Path(f"../prompts_and_responses/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            f"../prompts_and_responses/{retrieval_method_name}/{name}.json",
            "w",
        ) as f:
            for p, r in zip(prompts, responses):
                json.dump({"prompt": p, "response": r}, f)
                f.write("\n")
        return result_file_name

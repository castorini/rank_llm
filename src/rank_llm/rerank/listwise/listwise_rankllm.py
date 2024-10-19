import copy
import json
import logging
import random
import re
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

from ftfy import fix_text
from tqdm import tqdm

from rank_llm.data import RankingExecInfo, Request, Result
from rank_llm.rerank import PromptMode, RankLLM
from rank_llm.rerank.listwise.reorder.reorder_policy import (
    ModelFunction,
    ReorderPolicy,
    SlidingWindowReorderPolicy,
)
from rank_llm.rerank.listwise.reorder.top_down_reorder_policy import (
    TopDownReorderPolicy,
)
from rank_llm.rerank.listwise.reorder.tournament_sort_reorder_policy import (
    TournamentSortReorderPolicy,
)

logger = logging.getLogger(__name__)

SUPPORT_REORDER_POLICIES = [
    SlidingWindowReorderPolicy,
    TournamentSortReorderPolicy,
    TopDownReorderPolicy,
]


@dataclass
class RerankConsumption:
    consumption_reference_by_batch: int
    consumption_reference_by_item: int


class ListwiseRankLLM(RankLLM, ABC):
    """
    All children of ListwiseRankLLM must implement these functions:
        - rerank_batched
        - run_llm_batched
        - run_llm
        - create_prompt_batched
        - create_prompt
        - get_num_tokens
        - cost_per_1k_token
        - num_output_tokens
    """

    def __init__(
        self,
        reorder_policy: ReorderPolicy,
        model: str,
        context_size: int,
        window_size: int,
        prompt_mode: PromptMode,
        num_few_shot_examples: int,
    ) -> None:
        super().__init__(model, context_size, prompt_mode)
        self._num_few_shot_examples = num_few_shot_examples

        self.reorder_policy = (
            SlidingWindowReorderPolicy() if reorder_policy is None else reorder_policy
        )
        self._window_size = window_size

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        batched: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        populate_exec_summary: bool = kwargs.get("populate_exec_summary", False)

        batch_size = kwargs.get("batch_size", 1)

        if not batched:
            batch_size = 1

        reorder_policy = self.reorder_policy
        model_functions, consumption = self._get_model_function(batched, **kwargs)

        # reranking using vllm
        if len(set([len(req.candidates) for req in requests])) != 1:
            raise ValueError("Batched requests must have the same number of candidates")

        result: list[Result] = []

        with tqdm(range(0, len(requests)), leave=False) as bar:
            for i in range(0, len(requests), batch_size):
                batch = requests[i : min(i + batch_size, len(requests))]
                batch_result = reorder_policy.reorder(
                    requests=[
                        Result(
                            query=copy.deepcopy(request.query),
                            candidates=copy.deepcopy(request.candidates),
                            ranking_exec_summary=[],
                        )
                        for request in batch
                    ],
                    rank_start=max(rank_start, 0),
                    rank_end=min(
                        rank_end, len(requests[0].candidates)
                    ),  # TODO: Fails arbitrary hit sizes
                    model=model_functions,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_exec_summary=populate_exec_summary,
                )
                result.extend(batch_result)
                bar.update(len(batch))

        logger.info(
            f"Average consumption per request: {consumption.consumption_reference_by_item / len(requests) : .2f}"
        )

        return result

    def get_output_filename(
        self,
        top_k_candidates: int,
        dataset_name: str,
        shuffle_candidates: bool,
        **kwargs: Any,
    ) -> str:
        _modelname = self._model.split("/")[-1]
        if _modelname.startswith("checkpoint"):
            _modelname = self._model.split("/")[-2] + "_" + _modelname
        name = (
            f"{_modelname}_{self._context_size}_{top_k_candidates}_{self._prompt_mode}"
        )
        if dataset_name:
            name = f"{name}_{dataset_name}"
        if self._num_few_shot_examples > 0:
            name += f"_{self._num_few_shot_examples}_shot"
        name += f"_{self.reorder_policy.param_name()}"
        return (
            f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
            if shuffle_candidates
            else f"{name}_{datetime.isoformat(datetime.now())}"
        )

    def max_tokens(self) -> int:
        """
        Returns the maximum number of tokens for a given model

        Returns:
            int: The maximum token count.
        """
        return self._context_size

    # @deprecated("old sliding window pipeline is deprecated. please use reorder policy")
    def permutation_pipeline_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        logging: bool = False,
        populate_exec_summary: bool = False,
    ) -> List[Result]:
        """
        Runs the permutation pipeline on a batch of result objects within the passed in rank range.

        Args:
            results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            List[Result]: The list of processed result objects after applying permutation.
        """
        prompts = []
        logger.info("Loading prompts.")
        prompts = self.create_prompt_batched(
            results,
            [list(range(rank_start, rank_end)) for _ in range(len(results))],
            batch_size=32,
        )
        if logging:
            for prompt in prompts:
                logger.debug(f"Prompt: {prompt[0]}\n")
        logger.info("Prompts loaded.")
        batched_results = self.run_llm_batched(
            [prompt for prompt, _ in prompts], current_window_size=rank_end - rank_start
        )

        for index, (result, (prompt, in_token_count)) in enumerate(
            zip(results, prompts)
        ):
            permutation, out_token_count = batched_results[index]
            if logging:
                logger.debug(f"output: {permutation}")
            if populate_exec_summary:
                if result.ranking_exec_summary is None:
                    result.ranking_exec_summary = []
                ranking_exec_info = RankingExecInfo(
                    prompt, permutation, in_token_count, out_token_count
                )
                result.ranking_exec_summary.append(ranking_exec_info)
            result = self.receive_permutation(result, permutation, rank_start, rank_end)

        return results

    # @deprecated("old sliding window pipeline is deprecated. please use reorder policy")
    def permutation_pipeline(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
        populate_exec_summary: bool = True,
    ) -> Result:
        """
        Runs the permutation pipeline on the passed in result set within the passed in rank range.

        Args:
            result (Result): The result object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The processed result object after applying permutation.
        """
        prompt, in_token_count = self.create_prompt(
            result, list(range(rank_start, rank_end))
        )
        if logging:
            logger.info(f"Prompt: {prompt}\n")
        permutation, out_token_count = self.run_llm(
            prompt, current_window_size=rank_end - rank_start
        )
        if logging:
            print(f"Output: {permutation}")
        if populate_exec_summary:
            ranking_exec_info = RankingExecInfo(
                prompt, permutation, in_token_count, out_token_count
            )
            result.ranking_exec_summary.append(ranking_exec_info)
        result = self.receive_permutation(result, permutation, rank_start, rank_end)
        return result

    def shuffle_and_rescore(
        rerank_results: List[Result], rank_start: int, rank_end: int
    ):
        """
        Shuffles candidates between rank_start and rank_end, and rescales scores based on new rank.

        Args:
            rerank_results (List[Result]): List of Result objects to process.
            rank_start (int): Start index for ranking.
            rank_end (int): End index for ranking.
        """
        for rerank_result in rerank_results:
            # Shuffle rerank_result hits between rank_start and rank_end
            rerank_result.candidates[rank_start:rank_end] = random.sample(
                rerank_result.candidates[rank_start:rank_end],
                len(rerank_result.candidates[rank_start:rank_end]),
            )
            # Rescore all candidates with 1/rank
            for i, cand in enumerate(rerank_result.candidates):
                cand["score"] = 1.0 / (i + 1)
                cand["rank"] = i + 1

    # @deprecated("old sliding window pipeline is deprecated. please use reorder policy")
    def sliding_windows_batched(
        self,
        requests: List[Request],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
        populate_exec_summary: bool = False,
    ) -> List[Result]:
        """
        Applies the sliding window algorithm to the reranking process for a batch of result objects.
        Args:
            requests (List[Request]): The list of request objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            List[Result]: The list of result objects after applying the sliding window technique.
        """
        rerank_results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                ranking_exec_summary=[],
            )
            for request in requests
        ]
        if shuffle_candidates:
            self.shuffle_and_rescore(rerank_results, rank_start, rank_end)
        end_pos = rank_end
        start_pos = rank_end - window_size

        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + step != rank_start:
            if logging:
                logger.info(f"start_pos: {start_pos}, end_pos: {end_pos}")
            start_pos = max(start_pos, rank_start)
            rerank_results = self.permutation_pipeline_batched(
                rerank_results, start_pos, end_pos, logging, populate_exec_summary
            )
            end_pos = end_pos - step
            start_pos = start_pos - step
        return rerank_results

    # @deprecated("old sliding window pipeline is deprecated. please use reorder policy")
    def sliding_windows(
        self,
        request: Request,
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
        populate_exec_summary: bool = True,
    ) -> Result:
        """
        Applies the sliding window algorithm to the reranking process.

        Args:
            request (Request): The request object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The result object after applying the sliding window technique.
        """
        rerank_result = Result(
            query=copy.deepcopy(request.query),
            candidates=copy.deepcopy(request.candidates),
            ranking_exec_summary=[],
        )
        if shuffle_candidates:
            self.shuffle_and_rescore([rerank_result], rank_start, rank_end)
        end_pos = rank_end
        start_pos = rank_end - window_size
        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_result = self.permutation_pipeline(
                rerank_result,
                start_pos,
                end_pos,
                logging,
                populate_exec_summary=populate_exec_summary,
            )
            end_pos = end_pos - step
            start_pos = start_pos - step
        return rerank_result

    def get_ranking_cost_upperbound(
        self, num_q: int, rank_start: int, rank_end: int, window_size: int, step: int
    ) -> Tuple[float, int]:
        """
        Calculates the upper bound of the ranking cost for a given set of parameters.

        Args:
            num_q (int): The number of queries.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the cost and the total number of tokens used (input tokens + output tokens).
        """
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
        retrieved_results: List[Request],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
    ) -> Tuple[float, int]:
        """
        Calculates the ranking cost based on actual token counts from generated prompts.

        Args:
            retrieved_results (List[Request]): A list of retrieved results for processing.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the calculated cost and the total number of tokens used (input tokens + output tokens).
        """
        input_token_count = 0
        output_token_count = 0
        # Go through the retrieval result using the sliding window and count the number of tokens for generated prompts.
        # This is an estimated cost analysis since the actual prompts' length will depend on the ranking.
        for result in tqdm(retrieved_results):
            end_pos = rank_end
            start_pos = rank_end - window_size
            while start_pos >= rank_start:
                start_pos = max(start_pos, rank_start)
                prompt, _ = self.create_prompt(result, list(range(start_pos, end_pos)))
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
                if len(new_response) == 0 or new_response[-1] != " ":
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
        self, result: Result, permutation: str, rank_start: int, rank_end: int
    ) -> Result:
        """
        Processes and applies a permutation to the ranking results.

        This function takes a permutation string, representing the new order of items,
        and applies it to a subset of the ranking results. It adjusts the ranks and scores in the
        'result' object based on this permutation.

        Args:
            result (Result): The result object containing the initial ranking results.
            permutation (str): A string representing the new order of items.
                            Each item in the string should correspond to a rank in the results.
            rank_start (int): The starting index of the range in the results to which the permutation is applied.
            rank_end (int): The ending index of the range in the results to which the permutation is applied.

        Returns:
            Result: The updated result object with the new ranking order applied.

        Note:
            This function assumes that the permutation string is a sequence of integers separated by spaces.
            Each integer in the permutation string corresponds to a 1-based index in the ranking results.
            The function first normalizes these to 0-based indices, removes duplicates, and then reorders
            the items in the specified range of the 'result.candidates' list according to the permutation.
            Items not mentioned in the permutation string remain in their original sequence but are moved after
            the permuted items.
        """

        # Parse and normalize the permutation indices
        response = self._clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)

        # Extract the relevant candidates and create a mapping for new order
        cut_range = copy.deepcopy(result.candidates[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]

        # Update candidates in the new order
        for j, x in enumerate(response):
            result.candidates[j + rank_start] = copy.deepcopy(cut_range[x])
            if result.candidates[j + rank_start].score is not None:
                result.candidates[j + rank_start].score = cut_range[j].score

        return result

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

    def _permutation_to_rank(self, perm_string: str, selected_indices: List[int]):
        perm = [
            int(x) - 1 for x in self._clean_response(perm_string).strip().split(" ")
        ]
        perm = [
            int(x)
            for x in self._remove_duplicate(perm)
            if 0 <= int(x) < len(selected_indices)
        ]
        perm = perm + [i for i in range(len(selected_indices)) if i not in perm]
        return perm

    def _get_model_function(
        self, batched: bool = False, silence: bool = False, **kwargs
    ) -> Tuple[ModelFunction, RerankConsumption]:
        # [(Request, SelectIndex)] -> [Prompt]

        consumption = RerankConsumption(0, 0)

        if batched:

            def create_prompt(batch: List[Tuple[Result, List[int]]]):
                return [
                    prompt
                    for prompt, _ in self.create_prompt_batched(
                        [result for result, selected_indices in batch],
                        [selected_indices for result, selected_indices in batch],
                        32,
                    )
                ]

            def execute(
                batch: List[Union[str, Dict[str, str]]],
                selected_indices_batch: List[List[int]],
            ):
                consumption.consumption_reference_by_batch += 1
                consumption.consumption_reference_by_item += len(batch)

                return [
                    self._permutation_to_rank(s, selected_indices)
                    for (s, _), selected_indices in zip(
                        self.run_llm_batched(batch, silence=silence, **kwargs),
                        selected_indices_batch,
                    )
                ]

        else:

            def create_prompt(batch: List[Tuple[Result, List[int]]]):
                return [
                    self.create_prompt(result, selected_indices)[0]
                    for result, selected_indices in batch
                ]

            def execute(
                batch: List[Union[str, Dict[str, str]]],
                selected_indices_batch: List[List[int]],
            ):
                consumption.consumption_reference_by_batch += 1
                consumption.consumption_reference_by_item += len(batch)

                return [
                    self._permutation_to_rank(
                        self.run_llm(x, silence=silence, **kwargs)[0], selected_indices
                    )
                    for x, selected_indices in zip(batch, selected_indices_batch)
                ]

        return (
            ModelFunction(
                create_prompt=create_prompt,
                execute=execute,
                window_size=self._window_size,
            ),
            consumption,
        )

    @staticmethod
    def get_reorder_policy(reorder_policy: str, **kwargs):
        for policy in SUPPORT_REORDER_POLICIES:
            if reorder_policy.startswith(policy.name()):
                reorder_params = reorder_policy[len(policy.name()) :]
                if len(reorder_params) <= 1:
                    return policy()
                else:
                    assert reorder_params[0] == ":" and reorder_params[1] == "{"
                    reorder_params = reorder_params[1:]
                    try:
                        reorder_param_dict = json.loads(reorder_params)
                        if not isinstance(reorder_param_dict, dict):
                            raise Exception()
                    except Exception as e:
                        print(e)
                        raise Exception(f"Cannot load reorder policy {reorder_policy}")
                    return policy(**reorder_param_dict, extra_args=dict(**kwargs))

        raise Exception(f"Cannot find reorder policy {reorder_policy}")

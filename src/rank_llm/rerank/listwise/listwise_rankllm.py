import copy
import logging
import random
from abc import ABC
from datetime import datetime
from typing import Any, List, Optional, Tuple

import torch
from tqdm import tqdm

from rank_llm.data import InferenceInvocation, Request, Result
from rank_llm.rerank.rankllm import PromptMode, RankLLM

logger = logging.getLogger(__name__)

ALPH_START_IDX = ord("A") - 1


class ListwiseRankLLM(RankLLM, ABC):
    """
    Abstract base class that all listwise rerankers inherit.

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
        model: str,
        context_size: int,
        prompt_mode: Optional[PromptMode] = None,
        prompt_template_path: Optional[str] = None,
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        window_size: int = 20,
        stride: int = 10,
        use_alpha: bool = False,
        batch_size: int = 32,
    ) -> None:
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
        )
        self._window_size = window_size
        self._device = device
        self._use_alpha = use_alpha
        self._batch_size = batch_size
        self._stride = stride

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
        name = f"{_modelname}_{self._context_size}_{top_k_candidates}"
        if dataset_name:
            name = f"{name}_{dataset_name}"
        if self._num_few_shot_examples > 0:
            name += f"_{self._num_few_shot_examples}_shot"
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

    def permutation_pipeline_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        logging: bool = False,
        populate_invocations_history: bool = False,
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
        prompts = self.create_prompt_batched(results, rank_start, rank_end)
        if logging:
            for prompt in prompts:
                logger.debug(f"Prompt: {prompt[0]}\n")
        logger.info("Prompts loaded.")
        batched_results = self.run_llm_batched(
            [prompt for prompt, _ in prompts], current_window_size=rank_end - rank_start
        )
        ## Legacy format (text, out_token_count)
        ## TODO: Remove this once all the listwise rerankers have switched to the new format.
        if len(batched_results[0]) == 2:
            for index, (result, (prompt, in_token_count)) in enumerate(
                zip(results, prompts)
            ):
                permutation, out_token_count = batched_results[index]
                if logging:
                    logger.debug(f"output: {permutation}")
                if populate_invocations_history:
                    if result.invocations_history is None:
                        result.invocations_history = []
                    inference_invocation = InferenceInvocation(
                        prompt,
                        permutation,
                        in_token_count,
                        out_token_count,
                        self._inference_handler.template["output_validation_regex"],
                        self._inference_handler.template["output_extraction_regex"],
                    )
                    result.invocations_history.append(inference_invocation)
        else:
            ## New format (text, reasoning, usage)
            assert len(batched_results[0]) == 3
            for index, (result, (prompt, in_token_count)) in enumerate(
                zip(results, prompts)
            ):
                permutation, reasoning, usage = batched_results[index]
                in_token_count = usage.get("prompt_tokens") or usage.get("input_tokens")
                out_token_count = usage.get("completion_tokens") or usage.get(
                    "output_tokens"
                )
                if logging:
                    logger.debug(f"output: {permutation}")
                if populate_invocations_history:
                    if result.invocations_history is None:
                        result.invocations_history = []
                    inference_invocation = InferenceInvocation(
                        prompt,
                        permutation,
                        in_token_count,
                        out_token_count,
                        reasoning=reasoning,
                        token_usage=usage,
                        output_validation_regex=self._inference_handler.template[
                            "output_validation_regex"
                        ],
                        output_extraction_regex=self._inference_handler.template[
                            "output_extraction_regex"
                        ],
                    )
                    result.invocations_history.append(inference_invocation)
        result = self.receive_permutation(
            result, permutation, rank_start, rank_end, logging
        )

        return results

    def permutation_pipeline(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
        populate_invocations_history: bool = True,
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
        prompt, in_token_count = self.create_prompt(result, rank_start, rank_end)
        if logging:
            logger.info(f"Prompt: {prompt}\n")
        llm_result = self.run_llm(prompt, current_window_size=rank_end - rank_start)
        ## Legacy format (text, out_token_count)
        if len(llm_result) == 2:
            permutation, out_token_count = llm_result
            if logging:
                print(f"Output: {permutation}")
            if populate_invocations_history:
                inference_invocation = InferenceInvocation(
                    prompt,
                    permutation,
                    in_token_count,
                    out_token_count,
                    self._inference_handler.template["output_validation_regex"],
                    self._inference_handler.template["output_extraction_regex"],
                )
                result.invocations_history.append(inference_invocation)
        else:
            ## New format (text, reasoning, usage)
            permutation, reasoning, usage = llm_result
            in_token_count = (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or in_token_count
            )
            out_token_count = (
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            if logging:
                print(f"Output: {permutation}")
            if populate_invocations_history:
                if result.invocations_history is None:
                    result.invocations_history = []
                inference_invocation = InferenceInvocation(
                    prompt,
                    permutation,
                    in_token_count,
                    out_token_count,
                    reasoning=reasoning,
                    token_usage=usage,
                    output_validation_regex=self._inference_handler.template[
                        "output_validation_regex"
                    ],
                    output_extraction_regex=self._inference_handler.template[
                        "output_extraction_regex"
                    ],
                )
                result.invocations_history.append(inference_invocation)
        result = self.receive_permutation(
            result, permutation, rank_start, rank_end, logging
        )
        return result

    def shuffle_and_rescore(
        self, rerank_results: List[Result], rank_start: int, rank_end: int
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
                cand.score = 1.0 / (i + 1)

    def sliding_windows_batched(
        self,
        requests: List[Request],
        rank_start: int,
        rank_end: int,
        top_k_retrieve: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
        populate_invocations_history: bool = False,
    ) -> List[Result]:
        """
        Applies the sliding window algorithm to the reranking process for a batch of result objects.
        Args:
            requests (List[Request]): The list of request objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            top_k_retrieve (int): The number of candidate documents retrieved from the first-stage retrieval system before reranking.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            List[Result]: The list of result objects after applying the sliding window technique.
        """
        stride = self._stride
        window_size = min(self._window_size, top_k_retrieve)
        rerank_results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                invocations_history=[],
            )
            for request in requests
        ]
        if shuffle_candidates:
            self.shuffle_and_rescore(rerank_results, rank_start, rank_end)
        end_pos = rank_end
        start_pos = rank_end - window_size

        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + stride != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + stride != rank_start:
            if logging:
                logger.info(f"start_pos: {start_pos}, end_pos: {end_pos}")
            start_pos = max(start_pos, rank_start)
            rerank_results = self.permutation_pipeline_batched(
                rerank_results,
                start_pos,
                end_pos,
                logging,
                populate_invocations_history,
            )
            end_pos = end_pos - stride
            start_pos = start_pos - stride
        return rerank_results

    def sliding_windows(
        self,
        request: Request,
        rank_start: int,
        rank_end: int,
        top_k_retrieve: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
        populate_invocations_history: bool = True,
    ) -> Result:
        """
        Applies the sliding window algorithm to the reranking process for a single result object.
        Args:
            request (Request): The request object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            top_k_retrieve (int): The number of candidate documents retrieved from the first-stage retrieval system before reranking.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            Result: The result object after applying the sliding window technique.
        """
        stride = self._stride
        window_size = min(self._window_size, top_k_retrieve)
        rerank_result = Result(
            query=copy.deepcopy(request.query),
            candidates=copy.deepcopy(request.candidates),
            invocations_history=[],
        )
        if shuffle_candidates:
            self.shuffle_and_rescore([rerank_result], rank_start, rank_end)
        end_pos = rank_end
        start_pos = rank_end - window_size
        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + stride != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + stride != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_result = self.permutation_pipeline(
                rerank_result,
                start_pos,
                end_pos,
                logging,
                populate_invocations_history=populate_invocations_history,
            )
            end_pos = end_pos - stride
            start_pos = start_pos - stride
        return rerank_result

    def get_ranking_cost_upperbound(
        self, num_q: int, rank_start: int, rank_end: int
    ) -> Tuple[float, int]:
        """
        Calculates the upper bound of the ranking cost for a given set of parameters.

        Args:
            num_q (int): The number of queries.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.

        Returns:
            Tuple[float, int]: A tuple object containing the cost and the total number of tokens used (input tokens + output tokens).
        """
        # For every prompt generated for every query assume the max context size is used.
        num_promt = (rank_end - rank_start - self._window_size) / self._stride + 1
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
    ) -> Tuple[float, int]:
        """
        Calculates the ranking cost based on actual token counts from generated prompts.

        Args:
            retrieved_results (List[Request]): A list of retrieved results for processing.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.

        Returns:
            Tuple[float, int]: A tuple object containing the calculated cost and the total number of tokens used (input tokens + output tokens).
        """
        input_token_count = 0
        output_token_count = 0
        window_size = self._window_size
        stride = self._stride
        # Go through the retrieval result using the sliding window and count the number of tokens for generated prompts.
        # This is an estimated cost analysis since the actual prompts' length will depend on the ranking.
        for result in tqdm(retrieved_results):
            end_pos = rank_end
            start_pos = rank_end - window_size
            while start_pos >= rank_start:
                start_pos = max(start_pos, rank_start)
                prompt, _ = self.create_prompt(result, start_pos, end_pos)
                input_token_count += self.get_num_tokens(prompt)
                end_pos = end_pos - stride
                start_pos = start_pos - stride
                output_token_count += self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def receive_permutation(
        self,
        result: Result,
        permutation: str,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
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
        # Extract the relevant candidates
        cut_range = copy.deepcopy(result.candidates[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        try:
            # Parse and normalize the permutation indices
            response = self._inference_handler._clean_response(
                permutation, use_alpha=self._use_alpha
            )
            response = [int(x) - 1 for x in response.split()]
            response = self._remove_duplicate(response)

            # Create a mapping for new order
            response = [ss for ss in response if ss in original_rank]
            response = response + [tt for tt in original_rank if tt not in response]
        except Exception as e:
            if logging:
                print(f"exception {e} happened while handling response {permutation}")
            response = original_rank

        # Update candidates in the new order
        for j, x in enumerate(response):
            result.candidates[j + rank_start] = copy.deepcopy(cut_range[x])
            if result.candidates[j + rank_start].score is not None:
                result.candidates[j + rank_start].score = cut_range[j].score

        return result

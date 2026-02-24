import asyncio
import copy
import logging
import random
from abc import ABC
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

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
        max_passage_words: int = 300,
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
        self._max_passage_words = max_passage_words

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

    async def run_llm_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        current_window_size: Optional[int] = None,
    ) -> Tuple:
        """
        Async wrapper around run_llm. Subclasses with native async backends
        (AsyncLLMEngine, AsyncOpenAI) should override this for true concurrency.
        The default implementation runs run_llm in a thread-pool executor so
        that synchronous backends (SGLang, TensorRT) don't block the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.run_llm(prompt, current_window_size)
        )

    def _apply_llm_output_to_result(
        self,
        result: Result,
        llm_out: Tuple,
        prompt: Union[str, List[Dict[str, str]]],
        in_token_count: int,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
        populate_invocations_history: bool = False,
    ) -> Result:
        """
        Applies a single LLM output to a result object: records invocation history
        (if requested) and applies the permutation.
        """
        if len(llm_out) == 2:
            # Legacy format (text, out_token_count)
            permutation, out_token_count = llm_out
            if logging:
                logger.debug(f"output: {permutation}")
            if populate_invocations_history:
                if result.invocations_history is None:
                    result.invocations_history = []
                result.invocations_history.append(
                    InferenceInvocation(
                        prompt,
                        permutation,
                        in_token_count,
                        out_token_count,
                        self._inference_handler.template["output_validation_regex"],
                        self._inference_handler.template["output_extraction_regex"],
                    )
                )
        else:
            # New format (text, reasoning, usage)
            assert len(llm_out) == 3
            permutation, reasoning, usage = llm_out
            in_token_count = (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or in_token_count
            )
            out_token_count = (
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            if logging:
                logger.debug(f"output: {permutation}")
            if populate_invocations_history:
                if result.invocations_history is None:
                    result.invocations_history = []
                result.invocations_history.append(
                    InferenceInvocation(
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
                )
        return self.receive_permutation(
            result, permutation, rank_start, rank_end, logging
        )

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

        Uses a streaming work-queue approach: prompts are created lazily in mini-batches of
        self._batch_size, so at most batch_size prompts live in RAM at any time. Requests are
        processed independently — as soon as one request finishes a window its next window is
        immediately enqueued, keeping the LLM fully utilised without waiting for a lagging peer.

        Args:
            requests (List[Request]): The list of request objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            top_k_retrieve (int): The number of candidate documents retrieved from the first-stage retrieval system before reranking.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
            populate_invocations_history (bool, optional): Whether to record invocation history.
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

        # Each work item is (result_idx, start_pos, end_pos).
        # Initialise with the first (rightmost) window for every request.
        work_queue: deque = deque()
        for idx in range(len(rerank_results)):
            end_pos = min(rank_end, len(rerank_results[idx].candidates))
            start_pos = max(end_pos - window_size, rank_start)
            # Mirror the original loop condition before clamping start_pos.
            if end_pos > start_pos:
                work_queue.append((idx, start_pos, end_pos))

        # Exact total inference calls: sum the window count for each request
        # individually, accounting for its actual number of candidates.
        def _count_windows(num_candidates: int) -> int:
            end_pos = min(rank_end, num_candidates)
            start_pos = max(end_pos - window_size, rank_start)
            count = 0
            prev_start_pos = None
            # prev_start_pos != rank_start is to prevent redundant windows (e.g. 0-20, followed by 0-10)
            while end_pos > start_pos and prev_start_pos != rank_start:
                count += 1
                prev_start_pos = start_pos
                end_pos -= stride
                start_pos -= stride
            return count

        total_work = sum(_count_windows(len(r.candidates)) for r in rerank_results)
        progress = tqdm(total=total_work, desc="Sliding windows")

        # Semaphore caps the number of concurrently in-flight LLM requests.
        # Each coroutine handles one (request, window) pair end-to-end:
        # create prompt → await LLM → apply permutation → enqueue next window.
        # Because all coroutines are gathered concurrently, the async LLM
        # backend (AsyncLLMEngine / AsyncOpenAI) sees all requests at once and
        # schedules them with continuous batching — no waiting for stragglers.
        semaphore = asyncio.Semaphore(self._batch_size)
        pending_tasks: set = set()

        async def _process_one(idx: int, s: int, e: int) -> None:
            async with semaphore:
                prompt, in_tok = self.create_prompt(rerank_results[idx], s, e)
                if logging:
                    logger.debug(f"[req {idx}] window [{s}, {e}] prompt: {prompt}\n")
                llm_out = await self.run_llm_async(prompt, e - s)
                rerank_results[idx] = self._apply_llm_output_to_result(
                    rerank_results[idx],
                    llm_out,
                    prompt,
                    in_tok,
                    s,
                    e,
                    logging,
                    populate_invocations_history,
                )
                progress.update(1)

                # Immediately spawn the next window for this request so it
                # can run concurrently with all other in-flight windows.
                next_end = e - stride
                next_start = s - stride
                next_start = max(next_start, rank_start)
                if next_end > next_start and s != rank_start:
                    task = asyncio.ensure_future(
                        _process_one(idx, next_start, next_end)
                    )
                    pending_tasks.add(task)
                    task.add_done_callback(pending_tasks.discard)

        async def _run_all() -> None:
            # Seed: one task per request for its first window.
            for idx, s, e in work_queue:
                task = asyncio.ensure_future(_process_one(idx, s, e))
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)
            # Wait until every task (including dynamically spawned ones) is done.
            while pending_tasks:
                await asyncio.wait(list(pending_tasks))

        asyncio.run(_run_all())
        progress.close()
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

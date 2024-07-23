from datetime import datetime
from pathlib import Path
from typing import List
from enum import Enum

from tqdm import tqdm

from rank_llm.data import DataWriter, Request, Result
from rank_llm.rerank.rankllm import RankLLM

class OperationMode(Enum):
    STANDARD = 0
    VLLM = 1
    T5 = 2
    
    @classmethod
    def from_int(cls, val):
        for mode in cls:
            if mode.value == val:
                return mode
        raise ValueError(f"{val} is not a valid {cls.__name__}")

class Reranker:
    def __init__(self, agent: RankLLM) -> None:
        self._agent = agent

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        window_size: int = 20,
        step: int = 10,
        shuffle_candidates: bool = False,
        logging: bool = False,
        operation_mode: OperationMode = OperationMode.STANDARD,
        populate_exec_summary: bool = True,
        batched: bool = False,
    ) -> List[Result]:
        """
        Reranks a list of requests using the RankLLM agent.

        This function applies a sliding window algorithm to rerank the results.
        Each window of results is processed by the RankLLM agent to obtain a new ranking.

        Args:
            requests (List[Request]): The list of requests. Each request has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            window_size (int, optional): The size of each sliding window. Defaults to 20.
            step (int, optional): The step size for moving the window. Defaults to 10.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            vllm_batched (bool, optional): Whether to use VLLM batched processing. Defaults to False.
            populate_exec_summary (bool, optional): Whether to populate the exec summary. Defaults to False.
            batched (bool, optional): Whether to use batched processing. Defaults to False.

        Returns:
            List[Result]: A list containing the reranked candidates.
        """

        if operation_mode == OperationMode.STANDARD:
            results = []
            for request in tqdm(requests):
                result = self._agent.sliding_windows(
                    request,
                    rank_start=max(rank_start, 0),
                    rank_end=min(rank_end, len(request.candidates)),
                    window_size=window_size,
                    step=step,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_exec_summary=populate_exec_summary,
                )
                results.append(result)
            return results
        elif operation_mode == OperationMode.VLLM:
            if len(set([len(req.candidates) for req in requests])) !=1:
                raise ValueError("Batched requests must have the same number of candidates")
            
            return self._agent.sliding_windows_batched(
                requests,
                rank_start=max(rank_start, 0),
                rank_end=min(
                    rank_end, len(requests[0].candidates)
                ),  # TODO: Fails arbitrary hit sizes
                window_size=window_size,
                step=step,
                shuffle_candidates=shuffle_candidates,
                logging=logging,
            )
        
        else: # T5 Operation mode
            if batched:
                for i in range(1, len(requests)):
                    assert len(requests[0]) == len(requests[i]), "Batched requests must have the same number of candidates"
                return self._agent.sliding_windows_batched(
                    requests,
                    rank_start=max(rank_start, 0),
                    rank_end=min(
                        rank_end, len(requests[0].candidates)
                    ),  # TODO: Fails arbitrary hit sizes
                    window_size=window_size,
                    step=step,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                )

    def rerank(
        self,
        request: Request,
        rank_start: int = 0,
        rank_end: int = 100,
        window_size: int = 20,
        step: int = 10,
        shuffle_candidates: bool = False,
        logging: bool = False,
    ) -> Result:
        """
        Reranks a request using the RankLLM agent.

        This function applies a sliding window algorithm to rerank the results.
        Each window of results is processed by the RankLLM agent to obtain a new ranking.

        Args:
            request (Request): The reranking request which has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            window_size (int, optional): The size of each sliding window. Defaults to 20.
            step (int, optional): The step size for moving the window. Defaults to 10.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.

        Returns:
            Result: the rerank result which contains the reranked candidates.
        """
        results = self.rerank_batch(
            requests=[request],
            rank_start=rank_start,
            rank_end=rank_end,
            window_size=window_size,
            step=step,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
        )
        return results[0]

    def write_rerank_results(
        self,
        retrieval_method_name: str,
        results: List[Result],
        shuffle_candidates: bool = False,
        top_k_candidates: int = 100,
        pass_ct: int = None,
        window_size: int = None,
        dataset_name: str = None,
        rerank_results_dirname: str = "rerank_results",
        ranking_execution_summary_dirname: str = "ranking_execution_summary",
    ) -> str:
        """
        Writes the reranked results to files in specified formats.

        This function saves the reranked results in both TREC Eval format and JSON format.
        A summary of the ranking execution is saved as well.

        Args:
            retrieval_method_name (str): The name of the retrieval method.
            results (List[Result]): The reranked results to be written.
            shuffle_candidates (bool, optional): Indicates if the candidates were shuffled. Defaults to False.
            top_k_candidates (int, optional): The number of top candidates considered. Defaults to 100.
            pass_ct (int, optional): Pass count, if applicable. Defaults to None.
            window_size (int, optional): The window size used in reranking. Defaults to None.
            dataset_name (str, optional): The name of the dataset used. Defaults to None.

        Returns:
            str: The file name of the saved reranked results in TREC Eval format.

        Note:
            The function creates directories and files as needed. The file names are constructed based on the
            provided parameters and the current timestamp to ensure uniqueness so there are no collisions.
        """
        _modelname = self._agent._model.split("/")[-1]
        if _modelname.startswith("checkpoint"):
            _modelname = self._agent._model.split("/")[-2] + "_" + _modelname
        name = f"{_modelname}_{self._agent._context_size}_{top_k_candidates}_{self._agent._prompt_mode}"
        if dataset_name:
            name = f"{name}_{dataset_name}"
        if self._agent._num_few_shot_examples > 0:
            name += f"_{self._agent._num_few_shot_examples}_shot"
        name = (
            f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
            if shuffle_candidates
            else f"{name}_{datetime.isoformat(datetime.now())}"
        )
        if window_size is not None:
            name += f"_window_{window_size}"
        if pass_ct is not None:
            name += f"_pass_{pass_ct}"
        # write rerank results
        writer = DataWriter(results)
        Path(f"{rerank_results_dirname}/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        result_file_name = (
            f"{rerank_results_dirname}/{retrieval_method_name}/{name}.txt"
        )
        writer.write_in_trec_eval_format(result_file_name)
        writer.write_in_jsonl_format(
            f"{rerank_results_dirname}/{retrieval_method_name}/{name}.jsonl"
        )
        # Write ranking execution summary
        Path(f"{ranking_execution_summary_dirname}/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        writer.write_ranking_exec_summary(
            f"{ranking_execution_summary_dirname}/{retrieval_method_name}/{name}.json"
        )
        return result_file_name

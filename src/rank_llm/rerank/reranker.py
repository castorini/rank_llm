import json
from datetime import datetime
from pathlib import Path
from typing import List

from tqdm import tqdm

from rank_llm.rerank.rankllm import RankLLM
from rank_llm.result import Result, ResultsWriter


class Reranker:
    def __init__(self, agent: RankLLM) -> None:
        self._agent = agent

    def rerank(
        self,
        retrieved_results: List[Result],
        rank_start: int = 0,
        rank_end: int = 100,
        window_size: int = 20,
        step: int = 10,
        shuffle_candidates: bool = False,
        logging: bool = False,
    ) -> List[Result]:
        """
        Reranks a list of retrieved results using the RankLLM agent.

        This function applies a sliding window algorithm to rerank the results.
        Each window of results is processed by the RankLLM agent to obtain a new ranking.

        Args:
            retrieved_results (List[Result]): The list of results to be reranked.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            window_size (int, optional): The size of each sliding window. Defaults to 20.
            step (int, optional): The step size for moving the window. Defaults to 10.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.

        Returns:
            List[Result]: A list containing the reranked results.
        """
        rerank_results = []
        for result in tqdm(retrieved_results):
            rerank_result = self._agent.sliding_windows(
                result,
                rank_start=max(rank_start, 0),
                rank_end=min(rank_end, len(result.hits)),
                window_size=window_size,
                step=step,
                shuffle_candidates=shuffle_candidates,
                logging=logging,
            )
            rerank_results.append(rerank_result)
        return rerank_results

    def write_rerank_results(
        self,
        retrieval_method_name: str,
        results: List[Result],
        shuffle_candidates: bool = False,
        top_k_candidates: int = 100,
        pass_ct: int = None,
        window_size: int = None,
        dataset_name: str = None,
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
        writer = ResultsWriter(results)
        Path(f"rerank_results/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        result_file_name = f"rerank_results/{retrieval_method_name}/{name}.txt"
        writer.write_in_trec_eval_format(result_file_name)
        writer.write_in_json_format(
            f"rerank_results/{retrieval_method_name}/{name}.json"
        )
        # Write ranking execution summary
        Path(f"ranking_execution_summary/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        writer.write_ranking_exec_summary(
            f"ranking_execution_summary/{retrieval_method_name}/{name}.json"
        )
        return result_file_name

from pathlib import Path
from typing import Any, List, Optional, Tuple

from rank_llm.data import DataWriter, Request, Result
from rank_llm.rerank import (
    PromptMode,
    RankLLM,
    get_azure_openai_args,
    get_openai_api_key,
)
from rank_llm.rerank.listwise import RankListwiseOSLLM, SafeOpenai
from rank_llm.rerank.listwise.rank_fid import RankFiDDistill, RankFiDScore
from rank_llm.rerank.pointwise.monot5 import MonoT5
from rank_llm.rerank.rankllm import RankLLM


class Reranker:
    def __init__(self, agent: Optional[RankLLM]) -> None:
        self._agent = agent

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
            sglang_batched (bool, optional): Whether to use SGLang batched processing. Defaults to False.
            tensorrt_batched (bool, optional): Whether to use TensorRT-LLM batched processing. Defaults to False.
            populate_exec_summary (bool, optional): Whether to populate the exec summary. Defaults to False.
            batched (bool, optional): Whether to use batched processing. Defaults to False.

        Returns:
            List[Result]: A list containing the reranked candidates.
        """
        return self._agent.rerank_batch(
            requests, rank_start, rank_end, shuffle_candidates, logging, **kwargs
        )

    def rerank(
        self,
        request: Request,
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
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
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            **kwargs,
        )
        return results[0]

    def write_rerank_results(
        self,
        retrieval_method_name: str,
        results: List[Result],
        shuffle_candidates: bool = False,
        top_k_candidates: int = 100,
        dataset_name: str = None,
        rerank_results_dirname: str = "rerank_results",
        ranking_execution_summary_dirname: str = "ranking_execution_summary",
        vllm_batched: bool = False,
        sglang_batched: bool = False,
        tensorrt_batched: bool = False,
        **kwargs,
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
            vllm_batched (bool, optional): Indicates if vLLM inference backend used. Defaults to False.
            sglang_batched (bool, optional): Indicates if SGLang inference backend used. Defaults to False.

        Returns:
            str: The file name of the saved reranked results in TREC Eval format.

        Note:
            The function creates directories and files as needed. The file names are constructed based on the
            provided parameters and the current timestamp to ensure uniqueness so there are no collisions.
        """
        pass_ct: Optional[int] = kwargs.get("pass_ct", None)
        window_size: Optional[int] = kwargs.get("window_size", None)

        name = self._agent.get_output_filename(
            top_k_candidates, dataset_name, shuffle_candidates, **kwargs
        )

        if window_size is not None:
            name += f"_window_{window_size}"
        if pass_ct is not None:
            name += f"_pass_{pass_ct}"

        # Add vllm or sglang to rerank result file name if they are used
        if vllm_batched:
            name += "_vllm"
        if sglang_batched:
            name += "_sglang"
        if tensorrt_batched:
            name += "_tensorrt"

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

    def get_agent(self) -> RankLLM:
        return self._agent

    def create_agent(
        model_path: str,
        default_agent: RankLLM,
        interactive: bool,
        **kwargs: Any,
    ) -> RankLLM:
        """Construct rerank agent

        Keyword arguments:
        argument -- description
        model_path -- name of model
        default_agent -- used for interactive mode to pass in a pre-instantiated agent to use
        interactive -- whether to run retrieve_and_rerank in interactive mode, used by the API

        Return: rerank agent -- Option<RankLLM>
        """
        use_azure_openai: bool = kwargs.get("use_azure_openai", False)
        vllm_batched: bool = kwargs.get("vllm_batched", False)

        if interactive and default_agent is not None:
            # Default rerank agent
            agent = default_agent
        elif "gpt" in model_path or use_azure_openai:
            # GPT based reranking models

            keys_and_defaults = [
                ("context_size", 4096),
                ("prompt_mode", PromptMode.RANK_GPT),
                ("num_few_shot_examples", 0),
                ("window_size", 20),
            ]
            [
                context_size,
                prompt_mode,
                num_few_shot_examples,
                window_size,
            ] = extract_kwargs(keys_and_defaults, **kwargs)

            openai_keys = get_openai_api_key()
            agent = SafeOpenai(
                model=model_path,
                context_size=context_size,
                prompt_mode=prompt_mode,
                window_size=window_size,
                num_few_shot_examples=num_few_shot_examples,
                keys=openai_keys,
                **(get_azure_openai_args() if use_azure_openai else {}),
            )
        elif "vicuna" in model_path or "zephyr" in model_path:
            # RankVicuna or RankZephyr model suite
            print(f"Loading {model_path} ...")

            model_full_paths = {
                "rank_zephyr": "castorini/rank_zephyr_7b_v1_full",
                "rank_vicuna": "castorini/rank_vicuna_7b_v1",
            }

            keys_and_defaults = [
                ("context_size", 4096),
                ("prompt_mode", PromptMode.RANK_GPT),
                ("num_few_shot_examples", 0),
                ("device", "cuda"),
                ("num_gpus", 1),
                ("variable_passages", False),
                ("window_size", 20),
                ("system_message", None),
                ("vllm_batched", False),
                ("sglang_batched", False),
                ("tensorrt_batched", False),
                ("use_logits", False),
                ("use_alpha", False),
            ]
            [
                context_size,
                prompt_mode,
                num_few_shot_examples,
                device,
                num_gpus,
                variable_passages,
                window_size,
                system_message,
                vllm_batched,
                sglang_batched,
                tensorrt_batched,
                use_logits,
                use_alpha,
            ] = extract_kwargs(keys_and_defaults, **kwargs)

            agent = RankListwiseOSLLM(
                model=(
                    model_full_paths[model_path]
                    if model_path in model_full_paths
                    else model_path
                ),
                name=model_path,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                device=device,
                num_gpus=num_gpus,
                variable_passages=variable_passages,
                window_size=window_size,
                system_message=system_message,
                vllm_batched=vllm_batched,
                sglang_batched=sglang_batched,
                tensorrt_batched=tensorrt_batched,
                use_logits=use_logits,
                use_alpha=use_alpha,
            )

            print(f"Completed loading {model_path}")
        elif "monot5" in model_path:
            # using monot5
            print(f"Loading {model_path} ...")

            model_full_paths = {"monot5": "castorini/monot5-3b-msmarco-10k"}

            keys_and_defaults = [
                ("prompt_mode", PromptMode.MONOT5),
                ("context_size", 512),
                ("device", "cuda"),
                ("batch_size", 64),
            ]
            [prompt_mode, context_size, device, batch_size] = extract_kwargs(
                keys_and_defaults, **kwargs
            )

            agent = MonoT5(
                model=(
                    model_full_paths[model_path]
                    if model_path in model_full_paths
                    else model_path
                ),
                prompt_mode=prompt_mode,
                context_size=context_size,
                device=device,
                batch_size=batch_size,
            )

        elif "lit5-distill" in model_path.lower():
            keys_and_defaults = [
                ("context_size", 150),
                ("prompt_mode", PromptMode.LiT5),
                ("num_few_shot_examples", 0),
                ("window_size", 20),
                ("precision", "bfloat16"),
                ("device", "cuda"),
                # reuse this parameter, but its not for "vllm", but only for "batched"
                ("vllm_batched", False),
            ]
            (
                context_size,
                prompt_mode,
                num_few_shot_examples,
                window_size,
                precision,
                device,
                vllm_batched,
            ) = extract_kwargs(keys_and_defaults, **kwargs)

            agent = RankFiDDistill(
                model=model_path,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                window_size=window_size,
                precision=precision,
                device=device,
                batched=vllm_batched,
            )
            print(f"Completed loading {model_path}")
        elif "lit5-score" in model_path.lower():
            keys_and_defaults = [
                ("context_size", 150),
                ("prompt_mode", PromptMode.LiT5),
                ("num_few_shot_examples", 0),
                ("window_size", 100),
                ("precision", "bfloat16"),
                ("device", "cuda"),
                # reuse this parameter, but its not for "vllm", but only for "batched"
                ("vllm_batched", False),
            ]
            (
                context_size,
                prompt_mode,
                num_few_shot_examples,
                window_size,
                precision,
                device,
                vllm_batched,
            ) = extract_kwargs(keys_and_defaults, **kwargs)

            agent = RankFiDScore(
                model=model_path,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                window_size=window_size,
                precision=precision,
                device=device,
                batched=vllm_batched,
            )
            print(f"Completed loading {model_path}")
        elif vllm_batched:
            # supports loading models from huggingface
            print(f"Loading {model_path} ...")
            keys_and_defaults = [
                ("context_size", 4096),
                ("prompt_mode", PromptMode.RANK_GPT),
                ("num_few_shot_examples", 0),
                ("device", "cuda"),
                ("num_gpus", 1),
                ("variable_passages", False),
                ("window_size", 20),
                ("system_message", None),
                ("vllm_batched", True),
                ("use_logits", False),
                ("use_alpha", False),
            ]
            [
                context_size,
                prompt_mode,
                num_few_shot_examples,
                device,
                num_gpus,
                variable_passages,
                window_size,
                system_message,
                vllm_batched,
                use_logits,
                use_alpha,
            ] = extract_kwargs(keys_and_defaults, **kwargs)

            agent = RankListwiseOSLLM(
                model=(model_path),
                name=model_path,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                device=device,
                num_gpus=num_gpus,
                variable_passages=variable_passages,
                window_size=window_size,
                system_message=system_message,
                use_logits=use_logits,
                use_alpha=use_alpha,
                vllm_batched=vllm_batched,
            )

            print(f"Completed loading {model_path}")
        elif model_path in ["unspecified", "rank_random", "rank_identity"]:
            # NULL reranker
            agent = None
        else:
            raise ValueError(f"Unsupported model: {model_path}")

        if agent is None and model_path not in [
            "unspecified",
            "rank_random",
            "rank_identity",
        ]:
            raise ValueError(f"Unsupported model: {model_path}")
        return agent


def extract_kwargs(
    keys_and_defaults: List[Tuple[str, Any]],
    **kwargs,
) -> List[Any]:
    """Extract specified kwargs from **kwargs

    Keyword arguments:
    keys_and_defaults -- List of Tuple(keyname, default)
    Return: List of extracted kwargs in order provided in keys_and_defaults
    """
    extracted_kwargs = []
    for key, default in keys_and_defaults:
        value = kwargs.get(key, None)
        if value is None:
            value = default
        if (
            value is not None
            and default is not None
            and not isinstance(value, type(default))
        ):
            raise ValueError(
                f"Provided kwarg for {key} must be of type {type(default).__name__}, got {type(value).__name__}"
            )
        extracted_kwargs.append(value)
    return extracted_kwargs

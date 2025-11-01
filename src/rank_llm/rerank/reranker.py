from pathlib import Path
from typing import Any, List, Optional, Tuple

from rank_llm.data import DataWriter, Request, Result
from rank_llm.rerank import (
    RankLLM,
    get_azure_openai_args,
    get_genai_api_key,
    get_openai_api_key,
    get_openrouter_api_key,
)
from rank_llm.rerank.listwise import RankListwiseOSLLM, SafeGenai, SafeOpenai
from rank_llm.rerank.listwise.rank_fid import RankFiDDistill, RankFiDScore
from rank_llm.rerank.pairwise.duot5 import DuoT5
from rank_llm.rerank.pointwise.monoelectra import MonoELECTRA
from rank_llm.rerank.pointwise.monot5 import MonoT5
from rank_llm.rerank.rankllm import RankLLM


class Reranker:
    def __init__(self, model_coordinator: Optional[RankLLM]) -> None:
        self._model_coordinator = model_coordinator

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
        Reranks a list of requests using the RankLLM model_coordinator.

        This function applies a sliding window algorithm to rerank the results.
        Each window of results is processed by the RankLLM model_coordinator to obtain a new ranking.

        Args:
            requests (List[Request]): The list of requests. Each request has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            batched (bool, optional): Whether to use batched processing. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window_size.
        Returns:
            List[Result]: A list containing the reranked candidates.
        """
        return self._model_coordinator.rerank_batch(
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
        Reranks a request using the RankLLM model_coordinator.

        This function applies a sliding window algorithm to rerank the results.
        Each window of results is processed by the RankLLM model_coordinator to obtain a new ranking.

        Args:
            request (Request): The reranking request which has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window size.
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
        inference_invocations_history_dirname: str = "inference_invocations_history",
        sglang_batched: bool = False,
        tensorrt_batched: bool = False,
        **kwargs,
    ) -> str:
        """
        Writes the reranked results to files in specified formats.

        This function saves the reranked results in both TREC Eval format and JSON format.
        The history of inference invocations is saved as well.

        Args:
            retrieval_method_name (str): The name of the retrieval method.
            results (List[Result]): The reranked results to be written.
            shuffle_candidates (bool, optional): Indicates if the candidates were shuffled. Defaults to False.
            top_k_candidates (int, optional): The number of top candidates considered. Defaults to 100.
            pass_ct (int, optional): Pass count, if applicable. Defaults to None.
            window_size (int, optional): The window size used in reranking. Defaults to None.
            dataset_name (str, optional): The name of the dataset used. Defaults to None.
            sglang_batched (bool, optional): Indicates if SGLang inference backend used. Defaults to False.

        Returns:
            str: The file name of the saved reranked results in TREC Eval format.

        Note:
            The function creates directories and files as needed. The file names are constructed based on the
            provided parameters and the current timestamp to ensure uniqueness so there are no collisions.
        """
        pass_ct: Optional[int] = kwargs.get("pass_ct", None)
        window_size: Optional[int] = kwargs.get("window_size", None)

        name = self._model_coordinator.get_output_filename(
            top_k_candidates, dataset_name, shuffle_candidates, **kwargs
        )

        if window_size is not None:
            name += f"_window_{window_size}"
        if pass_ct is not None:
            name += f"_pass_{pass_ct}"

        # Add vllm or sglang or tensorrt to rerank result file name if they are used
        if sglang_batched:
            name += "_sglang"
        elif tensorrt_batched:
            name += "_tensorrt"
        else:
            # VLLM is the fallback right now
            name += "_vllm"

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
        # Write the history of inference invocations
        Path(f"{inference_invocations_history_dirname}/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        writer.write_inference_invocations_history(
            f"{inference_invocations_history_dirname}/{retrieval_method_name}/{name}.json"
        )
        return result_file_name

    def get_model_coordinator(self) -> RankLLM:
        return self._model_coordinator

    def create_model_coordinator(
        model_path: str,
        default_model_coordinator: RankLLM,
        interactive: bool,
        **kwargs: Any,
    ) -> RankLLM:
        """Construct rerank model_coordinator

        Keyword arguments:
        argument -- description
        model_path -- name of model
        default_model_coordinator -- used for interactive mode to pass in a pre-instantiated model_coordinator to use
        interactive -- whether to run retrieve_and_rerank in interactive mode, used by the API

        Return: rerank model_coordinator -- Option<RankLLM>
        """
        use_azure_openai: bool = kwargs.get("use_azure_openai", False)
        use_openrouter: bool = kwargs.get("use_openrouter", False)
        base_url: Optional[str] = kwargs.get("base_url")

        if interactive and default_model_coordinator is not None:
            # Default rerank model_coordinator
            model_coordinator = default_model_coordinator
        elif use_openrouter:
            keys_and_defaults = [
                ("context_size", 4096),
                (
                    "prompt_template_path",
                    "src/rank_llm/rerank/prompt_templates/rank_gpt_template.yaml",
                ),
                ("num_few_shot_examples", 0),
                ("few_shot_file", None),
                ("window_size", 20),
                ("stride", 10),
                ("batch_size", 32),
            ]
            [
                context_size,
                prompt_template_path,
                num_few_shot_examples,
                few_shot_file,
                window_size,
                stride,
                batch_size,
            ] = extract_kwargs(keys_and_defaults, **kwargs)
            openrouter_keys = get_openrouter_api_key()
            model_coordinator = SafeOpenai(
                model=model_path,
                context_size=context_size,
                prompt_template_path=prompt_template_path,
                window_size=window_size,
                stride=stride,
                num_few_shot_examples=num_few_shot_examples,
                few_shot_file=few_shot_file,
                batch_size=batch_size,
                keys=openrouter_keys,
                base_url="https://openrouter.ai/api/v1/",
                **(get_azure_openai_args() if use_azure_openai else {}),
            )
        elif "gpt" in model_path or use_azure_openai or base_url:
            # GPT based reranking models

            keys_and_defaults = [
                ("context_size", 4096),
                (
                    "prompt_template_path",
                    "src/rank_llm/rerank/prompt_templates/rank_gpt_template.yaml",
                ),
                ("num_few_shot_examples", 0),
                ("few_shot_file", None),
                ("window_size", 20),
                ("stride", 10),
                ("batch_size", 32),
                ("base_url", None),
            ]
            [
                context_size,
                prompt_template_path,
                num_few_shot_examples,
                few_shot_file,
                window_size,
                stride,
                batch_size,
                base_url,
            ] = extract_kwargs(keys_and_defaults, **kwargs)

            openai_keys = get_openai_api_key()
            model_coordinator = SafeOpenai(
                model=model_path,
                context_size=context_size,
                prompt_template_path=prompt_template_path,
                window_size=window_size,
                stride=stride,
                num_few_shot_examples=num_few_shot_examples,
                few_shot_file=few_shot_file,
                batch_size=batch_size,
                keys=openai_keys,
                base_url=base_url,
                **(get_azure_openai_args() if use_azure_openai else {}),
            )
        elif "gemini" in model_path:
            keys_and_defaults = [
                ("context_size", 4096),
                (
                    "prompt_template_path",
                    "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
                ),
                ("num_few_shot_examples", 0),
                ("few_shot_file", None),
                ("window_size", 20),
                ("stride", 10),
                ("batch_size", 32),
            ]
            [
                context_size,
                prompt_template_path,
                num_few_shot_examples,
                few_shot_file,
                window_size,
                stride,
                batch_size,
            ] = extract_kwargs(keys_and_defaults, **kwargs)

            genai_keys = get_genai_api_key()
            model_coordinator = SafeGenai(
                model=model_path,
                context_size=context_size,
                prompt_template_path=prompt_template_path,
                num_few_shot_examples=num_few_shot_examples,
                few_shot_file=few_shot_file,
                window_size=window_size,
                stride=stride,
                batch_size=batch_size,
                keys=genai_keys,
            )
        elif "monot5" in model_path:
            # using monot5
            print(f"Loading {model_path} ...")

            model_full_paths = {"monot5": "castorini/monot5-3b-msmarco-10k"}

            keys_and_defaults = [
                (
                    "prompt_template_path",
                    "src/rank_llm/rerank/prompt_templates/monot5_template.yaml",
                ),
                ("context_size", 512),
                ("num_few_shot_examples", 0),
                ("few_shot_file", None),
                ("device", "cuda"),
                ("batch_size", 32),
            ]
            [
                prompt_template_path,
                context_size,
                num_few_shot_examples,
                few_shot_file,
                device,
                batch_size,
            ] = extract_kwargs(keys_and_defaults, **kwargs)

            model_coordinator = MonoT5(
                model=(
                    model_full_paths[model_path]
                    if model_path in model_full_paths
                    else model_path
                ),
                prompt_template_path=prompt_template_path,
                context_size=context_size,
                num_few_shot_examples=num_few_shot_examples,
                few_shot_file=few_shot_file,
                device=device,
                batch_size=batch_size,
            )
        elif "monoelectra" in model_path.lower():
            # using monoelectra
            print(f"Loading {model_path} ...")

            model_full_paths = {"monoelectra": "crystina-z/monoELECTRA_LCE_nneg31"}

            keys_and_defaults = [
                (
                    "prompt_template_path",
                    "src/rank_llm/rerank/prompt_templates/monoelectra_template.yaml",
                ),
                ("context_size", 512),
                ("device", "cuda"),
                ("batch_size", 32),
            ]
            [
                prompt_template_path,
                context_size,
                device,
                batch_size,
            ] = extract_kwargs(keys_and_defaults, **kwargs)

            model_coordinator = MonoELECTRA(
                model=(
                    model_full_paths[model_path.lower()]
                    if model_path.lower() in model_full_paths
                    else model_path
                ),
                prompt_template_path=prompt_template_path,
                context_size=context_size,
                device=device,
                batch_size=batch_size,
            )
        elif "duot5" in model_path:
            # using duot5
            print(f"Loading {model_path} ...")

            model_full_paths = {"duot5": "castorini/duot5-3b-msmarco-10k"}

            keys_and_defaults = [
                (
                    "prompt_template_path",
                    "src/rank_llm/rerank/prompt_templates/duot5_template.yaml",
                ),
                ("context_size", 512),
                ("device", "cuda"),
                ("batch_size", 32),
            ]
            [
                prompt_template_path,
                context_size,
                device,
                batch_size,
            ] = extract_kwargs(keys_and_defaults, **kwargs)

            model_coordinator = DuoT5(
                model=(
                    model_full_paths[model_path]
                    if model_path in model_full_paths
                    else model_path
                ),
                prompt_template_path=prompt_template_path,
                context_size=context_size,
                device=device,
                batch_size=batch_size,
            )
        elif "lit5-distill" in model_path.lower():
            keys_and_defaults = [
                ("context_size", 150),
                (
                    "prompt_template_path",
                    "src/rank_llm/rerank/prompt_templates/rank_fid_template.yaml",
                ),
                ("window_size", 20),
                ("stride", 10),
                ("precision", "bfloat16"),
                ("device", "cuda"),
                ("batch_size", 32),
            ]
            (
                context_size,
                prompt_template_path,
                window_size,
                stride,
                precision,
                device,
                batch_size,
            ) = extract_kwargs(keys_and_defaults, **kwargs)

            model_coordinator = RankFiDDistill(
                model=model_path,
                context_size=context_size,
                prompt_template_path=prompt_template_path,
                window_size=window_size,
                stride=stride,
                precision=precision,
                device=device,
                batch_size=batch_size,
            )
            print(f"Completed loading {model_path}")
        elif "lit5-score" in model_path.lower():
            keys_and_defaults = [
                ("context_size", 150),
                (
                    "prompt_template_path",
                    "src/rank_llm/rerank/prompt_templates/rank_fid_score_template.yaml",
                ),
                ("window_size", 100),
                ("stride", 10),
                ("precision", "bfloat16"),
                ("device", "cuda"),
                ("batch_size", 32),
            ]
            (
                context_size,
                prompt_template_path,
                window_size,
                stride,
                precision,
                device,
                batch_size,
            ) = extract_kwargs(keys_and_defaults, **kwargs)

            model_coordinator = RankFiDScore(
                model=model_path,
                context_size=context_size,
                prompt_template_path=prompt_template_path,
                window_size=window_size,
                stride=stride,
                precision=precision,
                device=device,
                batch_size=batch_size,
            )
            print(f"Completed loading {model_path}")
        elif model_path in ["unspecified", "rank_random", "rank_identity"]:
            # NULL reranker
            model_coordinator = None
        else:
            # supports loading models from huggingface
            print(f"Loading {model_path} ...")
            model_full_paths = {
                "rank_zephyr": "castorini/rank_zephyr_7b_v1_full",
                "rank_vicuna": "castorini/rank_vicuna_7b_v1",
            }
            keys_and_defaults = [
                ("context_size", 4096),
                (
                    "prompt_template_path",
                    None,
                ),
                ("num_few_shot_examples", 0),
                ("few_shot_file", None),
                ("device", "cuda"),
                ("num_gpus", 1),
                ("variable_passages", False),
                ("window_size", 20),
                ("stride", 10),
                ("system_message", None),
                ("is_thinking", False),
                ("reasoning_token_budget", 10000),
                ("use_logits", False),
                ("use_alpha", False),
                ("batch_size", 32),
                ("base_url", None),
            ]
            [
                context_size,
                prompt_template_path,
                num_few_shot_examples,
                few_shot_file,
                device,
                num_gpus,
                variable_passages,
                window_size,
                stride,
                system_message,
                is_thinking,
                reasoning_token_budget,
                use_logits,
                use_alpha,
                batch_size,
                base_url,
            ] = extract_kwargs(keys_and_defaults, **kwargs)

            model_coordinator = RankListwiseOSLLM(
                model=(
                    model_full_paths[model_path]
                    if model_path in model_full_paths
                    else model_path
                ),
                name=model_path,
                context_size=context_size,
                prompt_template_path=prompt_template_path,
                num_few_shot_examples=num_few_shot_examples,
                few_shot_file=few_shot_file,
                device=device,
                num_gpus=num_gpus,
                variable_passages=variable_passages,
                window_size=window_size,
                stride=stride,
                system_message=system_message,
                is_thinking=is_thinking,
                reasoning_token_budget=reasoning_token_budget,
                use_logits=use_logits,
                use_alpha=use_alpha,
                batch_size=batch_size,
                base_url=base_url,
            )

            print(f"Completed loading {model_path}")

        if model_coordinator is None and model_path not in [
            "unspecified",
            "rank_random",
            "rank_identity",
        ]:
            raise ValueError(f"Unsupported model: {model_path}")
        return model_coordinator


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

import json
import os
from pathlib import Path
from typing import List, Union, Dict, Any

from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.reranker import Reranker
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod
from rank_llm.retrieve.retriever import Retriever, RetrievalMode
from rank_llm.retrieve.topics_dict import TOPICS


def get_api_key() -> str:
    return os.getenv("OPEN_AI_API_KEY")


def get_azure_openai_args() -> Dict[str, str]:
    azure_args = {
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
    }

    # Sanity check
    assert all(
        list(azure_args.values())
    ), "Ensure that `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION` are set"
    return azure_args


def retrieve_and_rerank(
    model_path: str,
    dataset: Union[str, List[str], List[Dict[str, Any]]],
    retrieval_mode: RetrievalMode,
    retrieval_method: RetrievalMethod,
    top_k_candidates: int = 100,
    context_size: int = 4096,
    device: str = "cuda",
    num_gpus: int = 1,
    prompt_mode: PromptMode = PromptMode.RANK_GPT,
    num_few_shot_examples: int = 0,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    query: str = "",
    use_azure_openai: bool = False,
    variable_passages: bool = False,
    num_passes: int = 1,
    window_size: int = 20,
    step_size: int = 10,
    system_message: str = None,
):
    # Construct Rerank Agent
    if "gpt" in model_path or use_azure_openai:
        from dotenv import dotenv_values, load_dotenv

        load_dotenv(dotenv_path=f".env.local")

        openai_keys = get_api_key()
        agent = SafeOpenai(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            keys=openai_keys,
            **(get_azure_openai_args() if use_azure_openai else {}),
        )
    elif "vicuna" in model_path.lower() or "zephyr" in model_path.lower():
        agent = RankListwiseOSLLM(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
            system_message=system_message,
        )
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    # Retrieve
    print("Retrieving:")
    if retrieval_mode == RetrievalMode.DATASET:
        candidates_file = Path(
            f"retrieve_results/{retrieval_method.name}/retrieve_results_{dataset}.json"
        )
        if not candidates_file.is_file():
            retriever = Retriever(RetrievalMode.DATASET)
            retriever.retrieve(dataset, retrieval_method=retrieval_method)
        else:
            print("Reusing existing retrieved results.")

        with open(candidates_file, "r") as f:
            retrieved_results = json.load(f)

    elif retrieval_mode == RetrievalMode.QUERY_AND_DOCUMENTS:
        retriever = Retriever(RetrievalMode.QUERY_AND_DOCUMENTS)
        retrieved_results = retriever.retrieve(dataset, query=query)

    elif retrieval_mode == RetrievalMode.QUERY_AND_HITS:
        retriever = Retriever(RetrievalMode.QUERY_AND_HITS)
        retrieved_results = retriever.retrieve(dataset, query=query)

    else:
        raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")
    print("Reranking:")
    reranker = Reranker(agent, top_k_candidates, dataset)
    for pass_ct in range(num_passes):
        print(f"Pass {pass_ct + 1} of {num_passes}:")
        (
            rerank_results,
            input_token_counts,
            output_token_counts,
            aggregated_prompts,
            aggregated_responses,
        ) = reranker.rerank(
            retrieved_results,
            rank_end=top_k_candidates,
            window_size=min(window_size, top_k_candidates),
            shuffle_candidates=shuffle_candidates,
            logging=print_prompts_responses,
            step=step_size,
        )

        # generate trec_eval file & evaluate for named datasets only
        if isinstance(dataset, str):
            file_name = reranker.write_rerank_results(
                retrieval_method.name,
                rerank_results,
                input_token_counts,
                output_token_counts,
                aggregated_prompts,
                aggregated_responses,
                shuffle_candidates,
                pass_ct=None if num_passes == 1 else pass_ct,
                window_size=window_size,
            )
            if (
                dataset in TOPICS
                and dataset not in ["dl22", "dl22-passage", "news"]
                and TOPICS[dataset] not in ["dl22", "dl22-passage", "news"]
            ):
                print("Evaluating:")
                EvalFunction.eval(
                    ["-c", "-m", "ndcg_cut.1", TOPICS[dataset], file_name]
                )
                EvalFunction.eval(
                    ["-c", "-m", "ndcg_cut.5", TOPICS[dataset], file_name]
                )
                EvalFunction.eval(
                    ["-c", "-m", "ndcg_cut.10", TOPICS[dataset], file_name]
                )
            else:
                print(f"Skipping evaluation as {dataset} is not in TOPICS.")
        if num_passes > 1:
            retrieved_results = rerank_results

    return rerank_results

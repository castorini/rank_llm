import copy
from typing import Any, Dict, List, Union

from rank_llm.data import Request, Query
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.api_keys import get_azure_openai_args, get_openai_api_key
from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.reranker import Reranker
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod
from rank_llm.retrieve.retriever import RetrievalMode, Retriever
from rank_llm.retrieve.service_retriever import ServiceRetriever
from rank_llm.retrieve.topics_dict import TOPICS


def retrieve_and_rerank(
    model_path: str,
    dataset: Union[str, List[str], List[Dict[str, Any]]],
    retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
    retrieval_method: RetrievalMethod = RetrievalMethod.BM25,
    top_k_retrieve: int = 50,
    top_k_rerank: int = 10,
    context_size: int = 4096,
    device: str = "cuda",
    num_gpus: int = 1,
    prompt_mode: PromptMode = PromptMode.RANK_GPT,
    num_few_shot_examples: int = 0,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    query: str = "",
    qid: int = 1,
    use_azure_openai: bool = False,
    variable_passages: bool = False,
    num_passes: int = 1,
    window_size: int = 20,
    step_size: int = 10,
    system_message: str = None,
    index_path: str = None,
    topics_path: str = None,
    index_type: str = None,
    interactive: bool = False,
    host: str = "http://localhost:8081",
):
    # Construct Rerank Agent
    model_full_path = ""        
    if "gpt" in model_path or use_azure_openai:
        openai_keys = get_openai_api_key()
        agent = SafeOpenai(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            keys=openai_keys,
            **(get_azure_openai_args() if use_azure_openai else {}),
        )
    elif model_path.lower()=="rank_zephyr" or model_path.lower()=="rank_vicuna":
        if model_path.lower()=="rank_zephyr":
            model_full_path="castorini/rank_zephyr_7b_v1_full"
        else:
            model_full_path= "castorini/rank_vicuna_7b_v1"
        agent = RankListwiseOSLLM(
            model=model_full_path,
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
    if interactive and retrieval_mode != RetrievalMode.DATASET: 
        raise ValueError(f"Unsupport retrieval mode for interactive retrieval. Currently only DATASET mode is supported.")
    
    if retrieval_mode == RetrievalMode.DATASET:
        if interactive:
            requests = [ServiceRetriever.from_dataset_with_prebuilt_index(
                dataset_name=dataset,
                k=top_k_retrieve,
                retrieval_method=retrieval_method, 
                host=host, request=Request(query=Query(text=query,qid=qid))
            )]
        else:
            requests = Retriever.from_dataset_with_prebuilt_index(
                dataset_name=dataset, retrieval_method=retrieval_method
            )

    elif retrieval_mode == RetrievalMode.CUSTOM:
        requests = Retriever.from_custom_index(
            index_path=index_path, topics_path=topics_path, index_type=index_type
        )
    else:
        raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")
    
    print("Reranking:")
    reranker = Reranker(agent)
    for pass_ct in range(num_passes):
        print(f"Pass {pass_ct + 1} of {num_passes}:")
        rerank_results = reranker.rerank_batch(
            requests,
            rank_end=top_k_rerank,
            window_size=min(window_size, top_k_rerank),
            shuffle_candidates=shuffle_candidates,
            logging=print_prompts_responses,
            step=step_size,
        )

        # generate trec_eval file & evaluate for named datasets only
        if isinstance(dataset, str):
            file_name = reranker.write_rerank_results(
                retrieval_method.name,
                rerank_results,
                shuffle_candidates,
                top_k_candidates=top_k_rerank,
                pass_ct=None if num_passes == 1 else pass_ct,
                window_size=window_size,
                dataset_name=dataset,
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
            requests = [
                Request(copy.deepycopy(r.query), copy.deepcopy(r.candidates))
                for r in rerank_results
            ]

    return rerank_results[0]

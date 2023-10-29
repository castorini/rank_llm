import json
from pathlib import Path
from typing import List, Union, Dict, Any

from rank_llm.rank_gpt import SafeOpenai
from rank_llm.rankllm import PromptMode
from rank_llm.rank_vicuna import RankVicuna
from rank_llm.trec_eval import EvalFunction
from rank_llm.pyserini_retriever import RetrievalMethod
from rank_llm.retriever import Retriever, RetrievalMode
from rank_llm.reranker import Reranker
from rank_llm.topics_dict import TOPICS


def get_api_key() -> str:
    from dotenv import dotenv_values, load_dotenv
    import os

    load_dotenv(dotenv_path=f".env.local")
    return os.getenv("OPEN_AI_API_KEY")


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
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    query: str = ""
):
    # Construct Rerank Agent
    if "gpt" in model_path:
        openai_keys = get_api_key()
        agent = SafeOpenai(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            keys=openai_keys,
        )
    else:
        agent = RankVicuna(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            device=device,
            num_gpus=num_gpus,
        )

    # Retrieve
    print("Retrieving:")
    if retrieval_mode == RetrievalMode.DATASET:
        candidates_file = Path(
            f"../retrieve_results/{retrieval_method.name}/retrieve_results_{dataset}.json"
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
    (
        rerank_results,
        input_token_counts,
        output_token_counts,
        aggregated_prompts,
        aggregated_responses
    ) = reranker.rerank(
        retrieved_results, 
        rank_end=top_k_candidates, 
        window_size=min(20, top_k_candidates),
        shuffle_candidates=shuffle_candidates,
        logging=print_prompts_responses)
    
    # generate trec_eval file & evaluate for named datasets only
    if isinstance(dataset, str):

        file_name = reranker.write_rerank_results(
            retrieval_method.name,
            rerank_results,
            input_token_counts,
            output_token_counts,
            aggregated_prompts,
            aggregated_responses,
            shuffle_candidates
        )

        print("Evaluating:")
        EvalFunction.eval(["-c", "-m", "ndcg_cut.1", TOPICS[dataset], file_name])
        EvalFunction.eval(["-c", "-m", "ndcg_cut.5", TOPICS[dataset], file_name])
        EvalFunction.eval(["-c", "-m", "ndcg_cut.10", TOPICS[dataset], file_name])

    return rerank_results

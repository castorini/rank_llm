import argparse
import json
from pathlib import Path
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch

from rank_llm.pyserini_retriever import RetrievalMethod
from rank_llm.rank_gpt import SafeOpenai
from rank_llm.rankllm import PromptMode
from rank_llm.rank_vicuna import RankVicuna
from rank_llm.topics_dict import TOPICS
from rank_llm.trec_eval import EvalFunction
from rank_llm.retrieve_and_rerank import RetrievalMode, Retriever, Reranker


def get_api_key() -> str:
    from dotenv import dotenv_values, load_dotenv
    import os

    load_dotenv(dotenv_path=f"../.env.local")
    return os.getenv("OPEN_AI_API_KEY")


def main(args):
    model_path = args.model_path
    context_size = args.context_size
    top_k_candidates = args.top_k_candidates
    dataset = args.dataset
    num_gpus = args.num_gpus
    retrieval_method = args.retrieval_method
    prompt_mode = args.prompt_mode
    shuffle_candidates = args.shuffle_candidates
    print_prompts_responses = args.print_prompts_responses
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    candidates_file = Path(
        f"../retrieve_results/{retrieval_method.name}/retrieve_results_{dataset}.json"
    )
    if not candidates_file.is_file():
        print("Retrieving:")
        retriever = Retriever(RetrievalMode.DATASET)
        retriever.retrieve(dataset=dataset, retrieval_method=retrieval_method)
    else:
        print("Reusing existing retrieved results.")

    with open(candidates_file, "r") as f:
        retrieved_results = json.load(f)

    print("Reranking:")
    reranker = Reranker(agent)
    (
        rerank_results,
        input_token_counts,
        output_token_counts,
        aggregated_prompts,
        aggregated_responses,
    ) = reranker.rerank(
        retrieved_results, 
        rank_end=top_k_candidates, 
        window_size=20,
        shuffle_candidates=shuffle_candidates,
        logging=print_prompts_responses)

    file_name = reranker.write_rerank_results(
        retrieval_method.name,
        rerank_results,
        input_token_counts,
        output_token_counts,
        aggregated_prompts,
        aggregated_responses,
        shuffle_candidates,
    )

    print("Evaluating:")
    EvalFunction.eval(["-c", "-m", "ndcg_cut.1", TOPICS[dataset], file_name])
    EvalFunction.eval(["-c", "-m", "ndcg_cut.5", TOPICS[dataset], file_name])
    EvalFunction.eval(["-c", "-m", "ndcg_cut.10", TOPICS[dataset], file_name])


""" sample run:
python rank_vicuna.py --model_path=../checkpoints/vicuna/vicuna-7b-checkpoint-800 --dataset=dl19 --retrieval_method=bm25
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--context_size", type=int, default=4096, help="context size used for model"
    )
    parser.add_argument(
        "--top_k_candidates",
        type=int,
        default=100,
        help="the number of top candidates to rerank",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"dataset name, must be in {TOPICS.keys()}",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="the number of GPUs to use"
    )
    parser.add_argument(
        "--retrieval_method",
        type=RetrievalMethod,
        required=True,
        choices=list(RetrievalMethod),
    )
    parser.add_argument(
        "--prompt_mode",
        type=PromptMode,
        required=True,
        choices=list(PromptMode),
    )
    parser.add_argument(
        "--shuffle_candidates",
        action="store_true",
        help="whether to shuffle the candidates before reranking",
    )
    parser.add_argument(
        "--print_prompts_responses",
        action="store_true",
        help="whether to print promps and responses",
    )
    args = parser.parse_args()
    main(args)

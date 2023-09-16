import argparse
from rank_llm import PromptMode
from rank_vicuna import RankVicuna
from rank_gpt import SafeOpenai
from pyserini_retriever import PyseriniRetriever, RetrievalMethod
from topics_dict import TOPICS
from pathlib import Path
from trec_eval import EvalFunction
import torch
from tqdm import tqdm
import json


def get_api_key():
    from dotenv import dotenv_values, load_dotenv
    import os

    load_dotenv(dotenv_path=f".env.local")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "gpt" in model_path:
        openai_keys = get_api_key()
        agent = SafeOpenai(
            model=model_path,
            context_size=context_size,
            top_k_candidates=top_k_candidates,
            dataset=dataset,
            prompt_mode=prompt_mode,
            keys=openai_keys,
        )
    else:
        agent = RankVicuna(
            model=model_path,
            context_size=context_size,
            top_k_candidates=top_k_candidates,
            dataset=dataset,
            prompt_mode=prompt_mode,
            device=device,
            num_gpus=num_gpus,
        )
    candidates_file = Path(
        f"retrieve_results/{retrieval_method.name}/retrieve_results_{dataset}.json"
    )
    if not candidates_file.is_file():
        print("Retrieving:")
        retriever = PyseriniRetriever(dataset, retrieval_method)
        # Always retrieve top 100 so that results are reusable for all top_k_candidates values.
        retriever.retrieve_and_store(k=100)
    else:
        print("Reusing existing retrieved results.")

    with open(candidates_file, "r") as f:
        retrieved_results = json.load(f)

    print("\nReranking:")
    rerank_results = []
    input_token_counts = []
    output_token_counts = []
    aggregated_prompts = []
    aggregated_responses = []
    for result in tqdm(retrieved_results):
        (
            rerank_result,
            in_token_count,
            out_token_count,
            prompts,
            responses,
        ) = agent.sliding_windows(
            result,
            rank_start=0,
            rank_end=top_k_candidates,
            window_size=20,
            step=10,
            shuffle_candidates=shuffle_candidates,
        )
        rerank_results.append(rerank_result)
        input_token_counts.append(in_token_count)
        output_token_counts.append(out_token_count)
        aggregated_prompts.extend(prompts)
        aggregated_responses.extend(responses)
    print(f"input_tokens_counts={input_token_counts}")
    print(f"total input token count={sum(input_token_counts)}")
    print(f"output_token_counts={output_token_counts}")
    print(f"total output token count={sum(output_token_counts)}")
    file_name = agent.write_rerank_results(
        retrieval_method.name,
        rerank_results,
        input_token_counts,
        output_token_counts,
        aggregated_prompts,
        aggregated_responses,
        shuffle_candidates,
    )
    EvalFunction.eval(["-c", "-m", "ndcg_cut.1", TOPICS[dataset], file_name])
    EvalFunction.eval(["-c", "-m", "ndcg_cut.5", TOPICS[dataset], file_name])
    EvalFunction.eval(["-c", "-m", "ndcg_cut.10", TOPICS[dataset], file_name])


""" sample run:
python rank_vicuna.py --model_path=checkpoints/vicuna/vicuna-7b-checkpoint-800 --dataset=dl19 --retrieval_method=bm25
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
    args = parser.parse_args()
    main(args)

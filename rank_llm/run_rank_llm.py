import argparse
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch

from rank_llm.pyserini_retriever import RetrievalMethod
from rank_llm.rankllm import PromptMode
from rank_llm.topics_dict import TOPICS
from rank_llm.retriever import RetrievalMode
from rank_llm.retrieve_and_rerank import retrieve_and_rerank


def main(args):
    model_path = args.model_path
    use_azure_openai = args.use_azure_openai
    context_size = args.context_size
    top_k_candidates = args.top_k_candidates
    dataset = args.dataset
    num_gpus = args.num_gpus
    retrieval_method = args.retrieval_method
    prompt_mode = args.prompt_mode
    num_few_shot_examples = args.num_few_shot_examples
    shuffle_candidates = args.shuffle_candidates
    print_prompts_responses = args.print_prompts_responses
    num_few_shot_examples = args.num_few_shot_examples
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variable_passages = args.variable_passages
    retrieval_mode = RetrievalMode.DATASET

    _ = retrieve_and_rerank(
        model_path,
        dataset,
        retrieval_mode,
        retrieval_method,
        top_k_candidates,
        context_size,
        device,
        num_gpus,
        prompt_mode,
        num_few_shot_examples,
        shuffle_candidates,
        print_prompts_responses,
        use_azure_openai=use_azure_openai,
        variable_passages=variable_passages,
    )


""" sample run:
python rank_llm/rank_vicuna.py --model_path=checkpoints/vicuna/vicuna-7b-checkpoint-800 --dataset=dl19 --retrieval_method=bm25
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model. If `use_azure_ai`, pass your deployment name."
    )
    parser.add_argument(
        "--use_azure_openai",
        action="store_true",
        help="If True, use Azure OpenAI. Requires env var to be set: "
            "`AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`"
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
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        required=False,
        default=0,
        help="number of in context examples to provide",
    )
    parser.add_argument(
        "--variable_passages",
        action="store_true",
        help="whether the model can account for variable number of passages in input",
    )
    args = parser.parse_args()
    main(args)

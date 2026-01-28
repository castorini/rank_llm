import argparse
import os
import sys
from pathlib import Path

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.rerank.rankllm import PromptMode
from rank_llm.retrieve import TOPICS, RetrievalMethod, RetrievalMode
from rank_llm.retrieve_and_rerank import retrieve_and_rerank


def main(args):
    model_path = args.model_path
    query = ""
    batch_size = args.batch_size
    use_azure_openai = args.use_azure_openai
    use_openrouter = args.use_openrouter
    base_url = args.base_url
    context_size = args.context_size
    top_k_candidates = args.top_k_candidates
    top_k_rerank = top_k_candidates if args.top_k_rerank == -1 else args.top_k_rerank
    max_queries = args.max_queries
    dataset = args.dataset
    num_gpus = args.num_gpus
    retrieval_method = args.retrieval_method
    requests_file = args.requests_file
    qrels_file = args.qrels_file
    output_jsonl_file = args.output_jsonl_file
    output_trec_file = args.output_trec_file
    invocations_history_file = args.invocations_history_file
    prompt_template_path = Path(args.prompt_template_path) if args.prompt_template_path else None
    num_few_shot_examples = args.num_few_shot_examples
    few_shot_file = args.few_shot_file
    shuffle_candidates = args.shuffle_candidates
    print_prompts_responses = args.print_prompts_responses
    num_few_shot_examples = args.num_few_shot_examples
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variable_passages = args.variable_passages
    retrieval_mode = (
        RetrievalMode.DATASET if args.dataset else RetrievalMode.CACHED_FILE
    )
    num_passes = args.num_passes
    stride = args.stride
    window_size = args.window_size
    system_message = args.system_message
    populate_invocations_history = args.populate_invocations_history
    is_thinking = args.is_thinking
    reasoning_token_budget = args.reasoning_token_budget
    use_logits = args.use_logits
    use_alpha = args.use_alpha
    sglang_batched = args.sglang_batched
    tensorrt_batched = args.tensorrt_batched
    reasoning_effort = args.reasoning_effort

    if args.requests_file:
        if args.retrieval_method:
            parser.error("--retrieval_method must not be used with --requests_file")

    if args.dataset and not args.retrieval_method:
        parser.error("--retrieval_method is required when --dataset is provided")

    _ = retrieve_and_rerank(
        model_path=model_path,
        query=query,
        batch_size=batch_size,
        dataset=dataset,
        retrieval_mode=retrieval_mode,
        requests_file=requests_file,
        qrels_file=qrels_file,
        output_jsonl_file=output_jsonl_file,
        output_trec_file=output_trec_file,
        invocations_history_file=invocations_history_file,
        retrieval_method=retrieval_method,
        top_k_retrieve=top_k_candidates,
        top_k_rerank=top_k_rerank,
        max_queries=max_queries,
        context_size=context_size,
        device=device,
        num_gpus=num_gpus,
        prompt_template_path=prompt_template_path,
        num_few_shot_examples=num_few_shot_examples,
        few_shot_file=few_shot_file,
        shuffle_candidates=shuffle_candidates,
        print_prompts_responses=print_prompts_responses,
        use_azure_openai=use_azure_openai,
        use_openrouter=use_openrouter,
        base_url=base_url,
        variable_passages=variable_passages,
        num_passes=num_passes,
        window_size=window_size,
        stride=stride,
        system_message=system_message,
        populate_invocations_history=populate_invocations_history,
        is_thinking=is_thinking,
        reasoning_token_budget=reasoning_token_budget,
        use_logits=use_logits,
        use_alpha=use_alpha,
        sglang_batched=sglang_batched,
        tensorrt_batched=tensorrt_batched,
        reasoning_effort=reasoning_effort,
    )


""" sample run:
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_vicuna_7b_v1  --top_k_candidates=100 --dataset=dl20  --retrieval_method=SPLADE++_EnsembleDistil_ONNX --prompt_mode=rank_GPT  --context_size=4096 --variable_passages
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model. If `use_azure_ai`, pass your deployment name.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Size of each batch for batched inference.",
    )
    parser.add_argument(
        "--use_azure_openai",
        action="store_true",
        help="If True, use Azure OpenAI. Requires env var to be set: "
        "`AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`",
    )
    parser.add_argument(
        "--use_openrouter",
        action="store_true",
        help="If True, use OpenRouter. Requires env var to be set: "
        "`OPENROUTER_API_KEY`",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="If using a non-OpenAI model, pass your base URL and provide API key. "
        "Requires env var to be set: `OPENAI_API_KEY`",
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
        "--top_k_rerank",
        type=int,
        default=-1,
        help="the number of top candidates to return from reranking",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="the max number of queries to process from the dataset",
    )
    retrieval_input_group = parser.add_mutually_exclusive_group(required=True)
    retrieval_input_group.add_argument(
        "--dataset",
        type=str,
        help=f"Should be one of 1- dataset name, must be in {TOPICS.keys()},  2- a list of inline documents  3- a list of inline hits; must be used when --requests_file is not specified",
    )
    parser.add_argument(
        "--retrieval_method",
        type=RetrievalMethod,
        help="Required if --dataset is used; must be omitted with --requests_file",
        choices=list(RetrievalMethod),
    )
    retrieval_input_group.add_argument(
        "--requests_file",
        type=str,
        help=f"Path to a JSONL file containing requests; must be used when --dataset is not specified.",
    )
    parser.add_argument(
        "--qrels_file",
        type=str,
        help="Only used with --requests_file; when present the Trec eval will be executed using this qrels file",
    )
    parser.add_argument(
        "--output_jsonl_file",
        type=str,
        help="Only used with --requests_file; when present, the ranked results will be saved in this JSONL file.",
    )
    parser.add_argument(
        "--output_trec_file",
        type=str,
        help="Only used with --requests_file; when present, the ranked results will be saved in this txt file in trec format.",
    )
    parser.add_argument(
        "--invocations_history_file",
        type=str,
        help="Only used with --requests_file and --populate_invocations_history; when present, the LLM invocations history (prompts, completions, and input/output token counts) will be stored in this file.",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="the number of GPUs to use"
    )
    parser.add_argument(
        "--prompt_mode",
        type=PromptMode,
        required=False,
        choices=list(PromptMode),
    )
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        required=False,
        help="yaml file path for the prompt template",
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
        "--few_shot_file",
        type=str,
        required=False,
        default=None,
        help="path to JSONL file containing few-shot examples.",
    )
    parser.add_argument(
        "--variable_passages",
        action="store_true",
        help="whether the model can account for variable number of passages in input",
    )
    parser.add_argument(
        "--num_passes",
        type=int,
        required=False,
        default=1,
        help="number of passes to run the model",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="window size for the sliding window approach",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="stride for the sliding window approach",
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
        help="the system message used in prompts",
    )
    parser.add_argument(
        "--populate_invocations_history",
        action="store_true",
        help="write a file with the prompts and raw responses from LLM",
    )
    parser.add_argument(
        "--is_thinking",
        action="store_true",
        help="enables thinking mode which increases output token budget to account for the full thinking trace + response.",
    )
    parser.add_argument(
        "--reasoning_token_budget",
        type=int,
        default=10000,
        help="number of output token budget for thinking traces on reasoning models",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="reasoning effort level for OpenAI reasoning models (e.g., o1, o3)",
    )
    infer_backend_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--use_logits",
        action="store_true",
        help="whether to rerank using the logits of the first identifier only.",
    )
    parser.add_argument(
        "--use_alpha",
        action="store_true",
        help="whether to use alphabetical identifers instead of numerical. Recommended when use_logits is True",
    )
    infer_backend_group.add_argument(
        "--sglang_batched",
        action="store_true",
        help="whether to run the model in batches using sglang backend",
    )
    infer_backend_group.add_argument(
        "--tensorrt_batched",
        action="store_true",
        help="whether to run the model in batches using tensorrtllm backend",
    )
    args = parser.parse_args()
    main(args)

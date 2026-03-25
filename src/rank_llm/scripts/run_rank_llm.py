import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.cli.operations import run_script_rerank
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.retrieve import TOPICS, RetrievalMethod

# Force spawn method to avoid "Cannot re-initialize CUDA in forked subprocess" error.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def main(args):
    parser_error = globals().get("parser")
    error_handler = parser_error.error if parser_error is not None else _raise_argument_error
    return run_script_rerank(args, parser_error=error_handler)


def _raise_argument_error(message):
    raise ValueError(message)


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
        help="Path to a JSONL file containing requests; must be used when --dataset is not specified.",
    )
    parser.add_argument(
        "--qrels_file",
        type=str,
        help="Optional. With --dataset: override default qrels. With --requests_file: qrels file for Trec eval.",
    )
    parser.add_argument(
        "--output_jsonl_file",
        type=str,
        help="Optional. With --dataset: override computed JSONL output path. With --requests_file: required path where ranked results are saved.",
    )
    parser.add_argument(
        "--output_trec_file",
        type=str,
        help="Optional. With --dataset: override computed TREC output path. With --requests_file: required path where ranked results are saved (trec format).",
    )
    parser.add_argument(
        "--invocations_history_file",
        type=str,
        help="Optional. With --dataset: override computed invocations history path. With --requests_file and --populate_invocations_history: required path for LLM invocations history (prompts, completions, and input/output token counts).",
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
    parser.add_argument(
        "--max_passage_words",
        type=int,
        default=300,
        help="maximum number of words per passage in the prompt (default: 300); in additon to truncating passages to this length, depending on the context size, window size, query length, etc further truncation may be needed.",
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

import argparse
import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.rerank import Reranker, get_genai_api_key
from rank_llm.rerank.listwise import SafeGenai
from rank_llm.retrieve import Retriever

TEMPLATES = files("rank_llm.rerank.prompt_templates")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank with Gemini via Google Gen AI.")
    parser.add_argument("--dataset", default="dl19")
    parser.add_argument(
        "--requests-file",
        default=None,
        help="Optional JSON/JSONL requests file. When set, skips retrieval.",
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-passage-words", type=int, default=300)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument(
        "--prompt-template-path",
        default=str(TEMPLATES / "rank_zephyr_template.yaml"),
    )
    parser.add_argument("--output-dir", default="demo_outputs")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not write inference invocation history.",
    )
    args = parser.parse_args()

    if args.requests_file:
        requests = read_requests_from_file(args.requests_file)
    else:
        requests = Retriever.from_dataset_with_prebuilt_index(args.dataset, k=args.k)
    if args.num_queries is not None:
        requests = requests[: args.num_queries]
    print(f"Loaded {len(requests)} requests from {args.dataset} (k={args.k}).")

    model_coordinator = SafeGenai(
        args.model,
        args.context_size,
        keys=get_genai_api_key(),
        prompt_template_path=args.prompt_template_path,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        max_passage_words=args.max_passage_words,
    )
    reranker = Reranker(model_coordinator)
    kwargs = {"populate_invocations_history": not args.no_history}
    rerank_results = reranker.rerank_batch(requests, **kwargs)
    print(rerank_results)

    writer = DataWriter(rerank_results)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    writer.write_in_jsonl_format(str(output_path / "rerank_results_gemini.jsonl"))
    writer.write_in_trec_eval_format(str(output_path / "rerank_results_gemini.txt"))
    if not args.no_history:
        writer.write_inference_invocations_history(
            str(output_path / "inference_invocations_history_gemini.json")
        )


if __name__ == "__main__":
    main()

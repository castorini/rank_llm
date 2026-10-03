import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(parent)

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.rerank.listwise import ZephyrReranker

DEFAULT_INPUT_FILE = "retrieve_results/BM25/retrieve_results_dl23_top20.jsonl"
DEFAULT_MODEL = "castorini/rank_zephyr_7b_v1_full"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerank stored first-stage retrieval results with RankZephyr."
    )
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help=f"Stored retrieval results in JSON or JSONL format (default: {DEFAULT_INPUT_FILE}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"RankZephyr model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Limit reranking to the first N queries (default: all).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Limit each query to the first K candidates (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="demo_outputs/stored_retrieved_results",
        help="Base output directory; results are nested under the model name.",
    )
    parser.add_argument(
        "--populate-invocations-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.num_queries is not None and args.num_queries <= 0:
        parser.error("--num-queries must be greater than 0")
    if args.k is not None and args.k <= 0:
        parser.error("--k must be greater than 0")
    if not 0 < args.stride <= args.window_size:
        parser.error("--stride must be greater than 0 and no larger than --window-size")

    requests = read_requests_from_file(args.input_file)
    if args.num_queries is not None:
        requests = requests[: args.num_queries]
    if not requests:
        parser.error("the input file contains no requests")
    if args.k is not None:
        for request in requests:
            request.candidates = request.candidates[: args.k]

    rank_end = args.k if args.k is not None else 100
    reranker = ZephyrReranker(
        model_path=args.model,
        context_size=args.context_size,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
    )
    try:
        rerank_results = reranker.rerank_batch(
            requests=requests,
            rank_end=rank_end,
            top_k_retrieve=rank_end,
            populate_invocations_history=args.populate_invocations_history,
        )
    finally:
        reranker.close()

    print(rerank_results)
    model_name = args.model.rsplit("/", 1)[-1].lower()
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = DataWriter(rerank_results)
    writer.write_in_jsonl_format(str(output_dir / "rerank.jsonl"))
    writer.write_in_trec_eval_format(str(output_dir / "rerank.txt"))
    history_path = output_dir / "invocations.json"
    if args.populate_invocations_history:
        writer.write_inference_invocations_history(str(history_path))
    else:
        history_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

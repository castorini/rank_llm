"""Retrieve a dataset with prebuilt indexes, rerank with RankVicuna, and evaluate.

Requires the pyserini and vllm extras, JDK 21, and a compatible GPU.
Example bounded run (retrieval still processes all topics):
    python src/rank_llm/demo/rerank_dataset_with_prebuilt_index.py \
        --retrieval-methods bm25 --num-queries 1 --k 4 --window-size 2 \
        --stride 1 --batch-size 1

Each retrieval method is evaluated (nDCG@10, MAP@100, R@20), reranked with the
same model, and evaluated again (nDCG@10).
Outputs are <output-dir>/<retrieval-method>/<model>/rerank.jsonl, rerank.txt,
and invocations.json; disabling history removes the previous invocations.json
in that directory.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(parent)

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import DataWriter
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.listwise import VicunaReranker
from rank_llm.retrieve import TOPICS, RetrievalMethod, Retriever

DEFAULT_DATASET = "dl19"
DEFAULT_RETRIEVAL_METHODS = [
    RetrievalMethod.BM25.value,
    RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL.value,
]
RETRIEVAL_METHOD_CHOICES = [
    method.value
    for method in RetrievalMethod
    if method not in (RetrievalMethod.UNSPECIFIED, RetrievalMethod.CUSTOM_INDEX)
]
DEFAULT_MODEL = "castorini/rank_vicuna_7b_v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve a dataset with prebuilt indexes and rerank with RankVicuna."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset to retrieve and rerank (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--retrieval-methods",
        nargs="+",
        choices=RETRIEVAL_METHOD_CHOICES,
        default=DEFAULT_RETRIEVAL_METHODS,
        help=f"Retrieval methods to run in order (default: {' '.join(DEFAULT_RETRIEVAL_METHODS)}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"RankVicuna model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of candidates to retrieve and rerank (default: 100).",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Limit reranking to the first N queries after retrieval (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="demo_outputs/prebuilt_index",
        help="Base output directory; outputs are nested under <retrieval-method>/<model>.",
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
    for name in (
        "k",
        "num_queries",
        "batch_size",
        "context_size",
        "window_size",
        "stride",
        "num_gpus",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    if not 0 < args.stride <= args.window_size:
        parser.error("--stride must be greater than 0 and no larger than --window-size")
    if args.dataset not in TOPICS:
        parser.error(f"unknown dataset: {args.dataset}")

    topics = TOPICS[args.dataset]
    model_name = args.model.rsplit("/", 1)[-1].lower()
    reranker = VicunaReranker(
        model_path=args.model,
        context_size=args.context_size,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
    )
    try:
        for method in args.retrieval_methods:
            retrieved_results = Retriever.from_dataset_with_prebuilt_index(
                args.dataset, RetrievalMethod(method), k=args.k
            )
            if args.num_queries is not None:
                retrieved_results = retrieved_results[: args.num_queries]
            if not retrieved_results:
                parser.error(f"dataset {args.dataset} has no queries")

            # Evaluate retrieved results. By default ndcg@10 is the eval metric,
            # other values can be specified.
            print(EvalFunction.from_results(retrieved_results, topics))
            # map_100
            eval_args = ["-c", "-m", "map_cut.100", "-l2"]
            print(EvalFunction.from_results(retrieved_results, topics, eval_args))
            # recall_20
            eval_args = ["-c", "-m", "recall.20"]
            print(EvalFunction.from_results(retrieved_results, topics, eval_args))

            rerank_results = reranker.rerank_batch(
                requests=retrieved_results,
                rank_end=args.k,
                top_k_retrieve=args.k,
                populate_invocations_history=args.populate_invocations_history,
            )

            # Analyze response. This reads the invocations history, so it needs
            # it populated.
            if args.populate_invocations_history:
                analyzer = ResponseAnalyzer.from_inline_results(rerank_results)
                print(analyzer.count_errors(verbose=True))

            # Evaluate rerank results.
            print(EvalFunction.from_results(rerank_results, topics))

            output_dir = Path(args.output_dir) / method / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            writer = DataWriter(rerank_results)
            writer.write_in_jsonl_format(str(output_dir / "rerank.jsonl"))
            writer.write_in_trec_eval_format(str(output_dir / "rerank.txt"))
            history_path = output_dir / "invocations.json"
            if args.populate_invocations_history:
                writer.write_inference_invocations_history(str(history_path))
            else:
                history_path.unlink(missing_ok=True)
    finally:
        reranker.close()


if __name__ == "__main__":
    main()

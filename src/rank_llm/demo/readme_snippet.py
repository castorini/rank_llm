"""Runnable version of the README walkthrough: retrieve, rerank, evaluate, analyze.

Requires the pyserini and vllm extras, JDK 21, and a compatible GPU.
Example bounded run (retrieval still processes all topics):
    python src/rank_llm/demo/readme_snippet.py \
        --num-queries 1 --k 4 --window-size 2 --stride 1 --batch-size 1

Outputs are <output-dir>/<model>/rerank.jsonl, rerank.txt, and invocations.json;
disabling history removes the previous invocations.json in that directory.
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

# from rank_llm.rerank import Reranker, get_openai_api_key
from rank_llm.rerank.listwise import (  # , SafeOpenai
    VicunaReranker,
    ZephyrReranker,
)
from rank_llm.retrieve.retriever import RetrievalMethod, Retriever
from rank_llm.retrieve.topics_dict import TOPICS

DEFAULT_DATASET = "dl19"
RETRIEVAL_METHOD_CHOICES = [
    method.value
    for method in RetrievalMethod
    if method not in (RetrievalMethod.UNSPECIFIED, RetrievalMethod.CUSTOM_INDEX)
]
DEFAULT_MODELS = {
    "zephyr": "castorini/rank_zephyr_7b_v1_full",
    "vicuna": "castorini/rank_vicuna_7b_v1",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve, rerank, evaluate, and analyze a dataset as in the README."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset to retrieve and rerank (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--retrieval-method",
        choices=RETRIEVAL_METHOD_CHOICES,
        default=RetrievalMethod.BM25.value,
        help=f"Retrieval method (default: {RetrievalMethod.BM25.value}).",
    )
    parser.add_argument(
        "--reranker",
        choices=tuple(DEFAULT_MODELS),
        default="zephyr",
        help="Reranker to run (default: zephyr).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model ID (default: {DEFAULT_MODELS['zephyr']} for zephyr, {DEFAULT_MODELS['vicuna']} for vicuna).",
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
        default="demo_outputs/readme_snippet",
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

    # ------ Retrieval ------

    # By default BM25 is used for retrieval of top 100 candidates.
    retrieved_results = Retriever.from_dataset_with_prebuilt_index(
        args.dataset, RetrievalMethod(args.retrieval_method), k=args.k
    )
    if args.num_queries is not None:
        retrieved_results = retrieved_results[: args.num_queries]
    if not retrieved_results:
        parser.error(f"dataset {args.dataset} has no queries")
    # -----------------------

    # ------- Rerank --------

    # Rank Zephyr or Rank Vicuna model
    model_path = args.model or DEFAULT_MODELS[args.reranker]
    reranker_class = ZephyrReranker if args.reranker == "zephyr" else VicunaReranker
    reranker = reranker_class(
        model_path=model_path,
        context_size=args.context_size,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
    )

    # RankGPT
    # model_coordinator = SafeOpenai("gpt-4o-mini", 4096, keys=get_openai_api_key())
    # reranker = Reranker(model_coordinator)

    try:
        rerank_results = reranker.rerank_batch(
            requests=retrieved_results,
            rank_end=args.k,
            top_k_retrieve=args.k,
            populate_invocations_history=args.populate_invocations_history,
        )
    finally:
        reranker.close()
    # -----------------------

    # ----- Evaluation ------

    # Evaluate retrieved results.
    topics = TOPICS[args.dataset]
    ndcg_10_retrieved = EvalFunction.from_results(retrieved_results, topics)
    print(ndcg_10_retrieved)

    # Evaluate rerank results.
    ndcg_10_rerank = EvalFunction.from_results(rerank_results, topics)
    print(ndcg_10_rerank)

    # By default ndcg@10 is the eval metric, other value can be specified:
    # eval_args = ["-c", "-m", "map_cut.100", "-l2"]
    # map_100_rerank = EvalFunction.from_results(rerank_results, topics, eval_args)
    # print(map_100_rerank)

    # eval_args = ["-c", "-m", "recall.20"]
    # recall_20_rerank = EvalFunction.from_results(rerank_results, topics, eval_args)
    # print(recall_20_rerank)

    # -----------------------

    # -- Analyze invocations ---
    # Response analysis reads the invocations history, so it needs it populated.
    if args.populate_invocations_history:
        analyzer = ResponseAnalyzer.from_inline_results(rerank_results)
        error_counts = analyzer.count_errors(verbose=True)
        print(error_counts)
    # -----------------------

    # ---- Save results ----
    model_name = model_path.rsplit("/", 1)[-1].lower()
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
    # -----------------------


if __name__ == "__main__":
    main()

"""Rerank BM25 results on BEIR datasets with ReasonRank and report nDCG@10.

Requires the pyserini and vllm extras, JDK 21, and a compatible GPU.
Example bounded run (retrieval still processes all topics):
    python src/rank_llm/demo/rerank_reasonrank_bm25_beir.py \
        --datasets scifact --num-queries 1 --k 4 --window-size 2 --stride 1 \
        --batch-size 1

By default each dataset uses batch size 1 for scifact, dbpedia, and nfc and 32
otherwise; --batch-size overrides this for every dataset.
Outputs are <output-dir>/<model>/<dataset>_bm25_top<k>.jsonl, .txt,
_invocations.json, and _metrics.json; disabling history removes the previous
_invocations.json for that dataset.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(parent)

from rank_llm.data import DataWriter
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM
from rank_llm.retrieve.retriever import Retriever
from rank_llm.retrieve.topics_dict import TOPICS

DEFAULT_DATASETS = ["scifact", "dbpedia", "nfc", "covid", "news", "signal", "robust04"]
# Some of the datasets have different number of retrieved candidates per query.
SINGLE_BATCH_DATASETS = ("scifact", "dbpedia", "nfc")
DEFAULT_MODEL = "liuwenhan/reasonrank-32B"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve BEIR datasets with BM25 and rerank them with ReasonRank."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to retrieve and rerank (default: {' '.join(DEFAULT_DATASETS)}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ReasonRank model ID (default: {DEFAULT_MODEL}).",
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
        help="Limit reranking to the first N queries per dataset after retrieval (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for every dataset (default: 1 for scifact, dbpedia, and nfc; 32 otherwise).",
    )
    parser.add_argument("--context-size", type=int, default=4096 * 8)
    parser.add_argument("--reasoning-token-budget", type=int, default=3072)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="rerank_results",
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
        "reasoning_token_budget",
        "window_size",
        "stride",
        "num_gpus",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    if not 0 < args.stride <= args.window_size:
        parser.error("--stride must be greater than 0 and no larger than --window-size")
    unknown = [dataset for dataset in args.datasets if dataset not in TOPICS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")

    templates = files("rank_llm.rerank.prompt_templates")
    model_name = args.model.rsplit("/", 1)[-1].lower()
    output_dir = Path(args.output_dir) / model_name
    for dataset in args.datasets:
        retrieve_results = Retriever.from_dataset_with_prebuilt_index(dataset, k=args.k)
        if args.num_queries is not None:
            retrieve_results = retrieve_results[: args.num_queries]
        if not retrieve_results:
            parser.error(f"dataset {dataset} has no queries")
        qrels = TOPICS[dataset]
        retrieve_ndcg_10 = EvalFunction.from_results(retrieve_results, qrels)
        batch_size = args.batch_size
        if batch_size is None:
            batch_size = 1 if dataset in SINGLE_BATCH_DATASETS else 32
        model_coordinator = RankListwiseOSLLM(
            context_size=args.context_size,
            model=args.model,
            is_thinking=True,
            reasoning_token_budget=args.reasoning_token_budget,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=batch_size,
            num_gpus=args.num_gpus,
            prompt_template_path=(templates / "reasonrank_template.yaml"),
        )
        try:
            rerank_results = Reranker(model_coordinator).rerank_batch(
                requests=retrieve_results,
                rank_end=args.k,
                top_k_retrieve=args.k,
                populate_invocations_history=args.populate_invocations_history,
            )
        finally:
            # vLLM runs the engine in a subprocess; close it before the next dataset.
            model_coordinator.close()

        rerank_ndcg_10 = EvalFunction.from_results(rerank_results, qrels)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / f"{dataset}_bm25_top{args.k}"
        writer = DataWriter(rerank_results)
        writer.write_in_jsonl_format(f"{prefix}.jsonl")
        writer.write_in_trec_eval_format(f"{prefix}.txt")
        history_path = Path(f"{prefix}_invocations.json")
        if args.populate_invocations_history:
            writer.write_inference_invocations_history(str(history_path))
        else:
            history_path.unlink(missing_ok=True)
        with open(f"{prefix}_metrics.json", "w") as f:
            json.dump({"retrieve": retrieve_ndcg_10, "rerank": rerank_ndcg_10}, f)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

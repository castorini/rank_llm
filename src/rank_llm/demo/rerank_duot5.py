"""
Pairwise reranking demo with DuoT5 (e.g. ``castorini/duot5-3b-msmarco-10k``)
on top of BM25 first-stage retrieval from a Pyserini prebuilt index.

DuoT5 is a T5 encoder-decoder pairwise reranker: for every pair of
candidates (d_i, d_j) it predicts whether d_i is more relevant than d_j
to the query, and an aggregation over those pairwise scores produces the
final ranking. The number of forward passes therefore grows quadratically
in k, so smaller k is recommended than for pointwise demos.

Prerequisites:
  pip install -e '.[pyserini,local]'

Usage (from repo root):
  python src/rank_llm/demo/rerank_duot5.py
  python src/rank_llm/demo/rerank_duot5.py \\
      --dataset dl19 \\
      --model castorini/duot5-base-msmarco \\
      --batch-size 16 \\
      --device cpu \\
      --num-queries 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, Result
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.pairwise.duot5 import DuoT5
from rank_llm.retrieve import TOPICS
from rank_llm.retrieve.retriever import Retriever

EVAL_METRICS: list[tuple[str, list[str]]] = [
    ("nDCG@10", ["-c", "-m", "ndcg_cut.10"]),
    ("MAP@100", ["-c", "-m", "map_cut.100", "-l2"]),
    ("Recall@20", ["-c", "-m", "recall.20"]),
    ("Recall@100", ["-c", "-m", "recall.100"]),
]


def _print_sample(results: list[Result], max_queries: int = 2, top_k: int = 5) -> None:
    for res in results[:max_queries]:
        print(f"qid={res.query.qid} text={res.query.text!r}")
        for c in res.candidates[:top_k]:
            print(f"  docid={c.docid} score={c.score:.4f}")


def _print_eval(results: list[Result], qrels: str) -> None:
    for label, eval_args in EVAL_METRICS:
        value = EvalFunction.from_results(results, qrels, eval_args)
        print(f"  {label:12s} {value}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pairwise DuoT5 reranking on BM25 first-stage results."
    )
    p.add_argument(
        "--dataset",
        default="dl20",
        help="TREC dataset name with a Pyserini prebuilt index (default: dl20).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=50,
        help="Top-k passages per query from first-stage retrieval (default: 50; "
        "pairwise scoring is quadratic in k).",
    )
    p.add_argument(
        "--model",
        default="castorini/duot5-3b-msmarco-10k",
        help="HuggingFace model id for DuoT5 (default: castorini/duot5-3b-msmarco-10k).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-step batch size for the pairwise scorer (default: 32).",
    )
    p.add_argument(
        "--context-size",
        type=int,
        default=512,
        help="Context size for the model (default: 512, matches DuoT5 default).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device to run the model on (default: cuda; use 'cpu' on machines without a GPU).",
    )
    p.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Cap the number of queries for a quick smoke test (default: all).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. When set, outputs go to "
        "{output_dir}/{model_tag}/{dataset}/.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip inline evaluation (useful in batch benchmark runs).",
    )
    args = p.parse_args()

    requests = Retriever.from_dataset_with_prebuilt_index(args.dataset, k=args.k)
    if args.num_queries is not None:
        requests = requests[: args.num_queries]
    print(f"Loaded {len(requests)} requests from {args.dataset} (k={args.k}).")

    qrels = TOPICS.get(args.dataset)
    run_eval = not args.skip_eval

    if qrels and run_eval:
        print(f"\n{'=' * 60}")
        print(f"Retrieval metrics  (BM25, k={args.k})")
        print(f"{'=' * 60}")
        _print_eval(requests, qrels)

    coordinator = DuoT5(
        model=args.model,
        context_size=args.context_size,
        device=args.device,
        batch_size=args.batch_size,
    )
    reranker = Reranker(coordinator)
    kwargs = {"populate_invocations_history": True}

    print(f"\n--- Batched reranking ({len(requests)} queries) ---")
    rerank_results = reranker.rerank_batch(requests, **kwargs)
    print(f"Reranked {len(rerank_results)} results.")
    _print_sample(rerank_results)

    if qrels and run_eval:
        print(f"\n{'=' * 60}")
        print(f"Reranking metrics  ({args.model})")
        print(f"{'=' * 60}")
        _print_eval(rerank_results, qrels)

    # --- Save outputs ---
    model_tag = args.model.split("/")[-1].lower()
    if args.output_dir:
        out_path = Path(args.output_dir) / model_tag / args.dataset
    else:
        out_path = Path("demo_outputs")
    out_path.mkdir(parents=True, exist_ok=True)

    writer = DataWriter(rerank_results)
    writer.write_in_jsonl_format(str(out_path / "rerank.jsonl"))
    writer.write_in_trec_eval_format(str(out_path / "rerank.txt"))
    writer.write_inference_invocations_history(str(out_path / "invocations.json"))


if __name__ == "__main__":
    main()

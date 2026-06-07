"""
Listwise reranking demo with LiT5 (e.g. ``castorini/LiT5-Distill-large`` and
``castorini/LiT5-Score-large``) on top of BM25 first-stage retrieval from a
Pyserini prebuilt index.

LiT5 is a Fusion-in-Decoder listwise reranker family with two variants:
  - **Distill**: encodes each (query, candidate) independently in the encoder
    then decodes a permutation over the candidates inside one window.
  - **Score**: same FiD encoder backbone but emits a per-candidate relevance
    score instead of a permutation. Useful when you want absolute scores or
    when downstream callers expect a calibrated value per document.

Like other listwise rerankers it uses a sliding window over the candidate
list (default window=20, stride=10).

Prerequisites:
  pip install -e '.[pyserini,local]'

Usage (from repo root):
  python src/rank_llm/demo/rerank_lit5.py
  python src/rank_llm/demo/rerank_lit5.py \\
      --variant distill \\
      --distill-model castorini/LiT5-Distill-base \\
      --dataset dl20 \\
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
from rank_llm.rerank.listwise.lit5_reranker import (
    LiT5DistillReranker,
    LiT5ScoreReranker,
)
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
        description="Listwise LiT5 reranking on BM25 first-stage results."
    )
    p.add_argument(
        "--variant",
        choices=("both", "distill", "score"),
        default="both",
        help="Which LiT5 variant to run (default: both — Distill across the "
        "full request set then a single-query Score demo, matching the "
        "original hardcoded script).",
    )
    p.add_argument(
        "--dataset",
        default="dl19",
        help="TREC dataset name with a Pyserini prebuilt index (default: dl19).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=100,
        help="Top-k passages per query from first-stage retrieval (default: 100).",
    )
    p.add_argument(
        "--distill-model",
        default="castorini/LiT5-Distill-large",
        help="HuggingFace model id for LiT5-Distill (default: castorini/LiT5-Distill-large).",
    )
    p.add_argument(
        "--score-model",
        default="castorini/LiT5-Score-large",
        help="HuggingFace model id for LiT5-Score (default: castorini/LiT5-Score-large).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-step batch size for the FiD scorer (default: 32).",
    )
    p.add_argument(
        "--context-size",
        type=int,
        default=300,
        help="Context size for the model (default: 300, matches LiT5 default).",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Sliding-window size over candidates (default: 20).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sliding-window stride (default: 10).",
    )
    p.add_argument(
        "--precision",
        default="bfloat16",
        help="Numerical precision for LiT5-Distill (default: bfloat16).",
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

    rerank_results: list[Result] = []
    run_distill = args.variant in ("both", "distill")
    run_score = args.variant in ("both", "score")

    if run_distill:
        distill = LiT5DistillReranker(
            model_path=args.distill_model,
            context_size=args.context_size,
            precision=args.precision,
            window_size=args.window_size,
            stride=args.stride,
            device=args.device,
            batch_size=args.batch_size,
        )
        kwargs = {
            "populate_invocations_history": True,
            "top_k_retrieve": args.k,
            "rank_end": args.k,
        }
        print(f"\n--- LiT5-Distill batched reranking ({len(requests)} queries) ---")
        rerank_results = distill.rerank_batch(requests, **kwargs)
        print(f"Reranked {len(rerank_results)} results.")
        _print_sample(rerank_results)

        if qrels and run_eval:
            print(f"\n{'=' * 60}")
            print(f"Reranking metrics  ({args.distill_model})")
            print(f"{'=' * 60}")
            _print_eval(rerank_results, qrels)

    if run_score:
        score = LiT5ScoreReranker(
            model_path=args.score_model,
            context_size=args.context_size,
            window_size=args.window_size,
            stride=args.stride,
            device=args.device,
            batch_size=args.batch_size,
        )
        request = requests[0]
        print(f"\n--- LiT5-Score single-query demo (qid={request.query.qid}) ---")
        score_results = score.rerank_batch(
            [request],
            populate_invocations_history=True,
            top_k_retrieve=args.k,
            rank_end=args.k,
        )
        score_result = score_results[0]
        print(f"qid={score_result.query.qid} text={score_result.query.text!r}")
        for c in score_result.candidates[:5]:
            print(f"  docid={c.docid} score={c.score:.4f}")
        if not rerank_results:
            rerank_results = score_results

    # --- Save outputs ---
    primary_model = args.distill_model if run_distill else args.score_model
    model_tag = primary_model.split("/")[-1].lower()
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

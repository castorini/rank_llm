"""
Demo: local encoder-based reranking with Jina Reranker v3, using the same
dl19 retrieval path as ``rerank_monot5.py`` / ``rerank_pointwise_vllm.py``
(BM25 + prebuilt index).

Jina Reranker v3 is a 0.6B-param model that scores up to 64 documents in
a single forward pass using causal cross-attention between query and
documents.  Unlike generative rerankers it runs locally via HuggingFace
``transformers`` — no server required.

Prerequisites:
  pip install -e '.[pyserini]'
  pip install transformers torch

Usage (from repo root, after indexes are available):

  python src/rank_llm/demo/rerank_jina.py

  python src/rank_llm/demo/rerank_jina.py \\
      --model jinaai/jina-reranker-v3 \\
      --window-size 32 \\
      --max-passage-words 512 \\
      --batch-size 2 \\
      --device cuda
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

from rank_llm.data import DataWriter
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.pointwise.jina_reranker import JinaReranker
from rank_llm.retrieve import TOPICS
from rank_llm.retrieve.retriever import Retriever


def _print_sample(results, max_queries: int = 2, top_k: int = 5) -> None:
    for res in results[:max_queries]:
        print(f"qid={res.query.qid} text={res.query.text!r}")
        for c in res.candidates[:top_k]:
            print(f"  docid={c.docid} score={c.score:.4f}")


EVAL_METRICS: list[tuple[str, list[str]]] = [
    ("nDCG@10", ["-c", "-m", "ndcg_cut.10"]),
    ("MAP@100", ["-c", "-m", "map_cut.100", "-l2"]),
    ("Recall@20", ["-c", "-m", "recall.20"]),
    ("Recall@100", ["-c", "-m", "recall.100"]),
]


def _print_eval(results, qrels: str) -> None:
    for label, eval_args in EVAL_METRICS:
        value = EvalFunction.from_results(results, qrels, eval_args)
        print(f"  {label:12s} {value}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rerank with Jina Reranker v3 (local HuggingFace model)."
    )
    p.add_argument(
        "--dataset",
        default="dl19",
        help="TREC dataset name with prebuilt index (default: dl19).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=100,
        help="Top-k passages per query from first-stage retrieval (default: 100).",
    )
    p.add_argument(
        "--model",
        default="jinaai/jina-reranker-v3",
        help="HuggingFace model ID (default: jinaai/jina-reranker-v3).",
    )
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--max-passage-words",
        type=int,
        default=512,
        help="Truncate each passage to this many words. "
        "When unset, derived from context_size / num_docs.",
    )
    p.add_argument("--context-size", type=int, default=131_072)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="auto")
    args = p.parse_args()

    requests = Retriever.from_dataset_with_prebuilt_index(args.dataset, k=args.k)
    print(f"Loaded {len(requests)} requests from {args.dataset} (k={args.k}).")

    qrels = TOPICS.get(args.dataset)

    # --- Retrieval-only evaluation ---
    if qrels:
        print(f"\n{'=' * 60}")
        print(f"Retrieval metrics  (BM25, k={args.k})")
        print(f"{'=' * 60}")
        _print_eval(requests, qrels)

    # --- Rerank ---
    print(f"\nLoading Jina model: {args.model} (device={args.device}) ...")
    coordinator = JinaReranker(
        model=args.model,
        context_size=args.context_size,
        device=args.device,
        window_size=args.window_size,
        batch_size=args.batch_size,
        dtype=args.dtype,
        max_passage_words=args.max_passage_words,
    )
    reranker = Reranker(coordinator)
    kwargs = {"populate_invocations_history": True}

    print("\n--- Reranking ---")
    results = reranker.rerank_batch(requests, **kwargs)
    print(f"Reranked {len(results)} queries.")
    _print_sample(results)

    # --- Reranking evaluation ---
    if qrels:
        print(f"\n{'=' * 60}")
        print(f"Reranking metrics  ({args.model})")
        print(f"{'=' * 60}")
        _print_eval(results, qrels)

    # --- Write outputs ---
    writer = DataWriter(results)
    Path("demo_outputs/").mkdir(parents=True, exist_ok=True)
    writer.write_in_jsonl_format("demo_outputs/rerank_results_jina.jsonl")
    writer.write_in_trec_eval_format("demo_outputs/rerank_results_jina.txt")
    writer.write_inference_invocations_history(
        "demo_outputs/inference_invocations_history_jina.json"
    )


if __name__ == "__main__":
    main()

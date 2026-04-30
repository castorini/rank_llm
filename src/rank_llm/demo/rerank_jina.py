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

  # Full dataset with evaluation
  python src/rank_llm/demo/rerank_jina.py

  # Custom model path / parameters
  python src/rank_llm/demo/rerank_jina.py \\
      --model jinaai/jina-reranker-v3 \\
      --batch-size 32 \\
      --max-passage-words 256 \\
      --device cuda

  # Quick smoke test with inline candidates (no pyserini required)
  python src/rank_llm/demo/rerank_jina.py --inline
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dacite import from_dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, Request
from rank_llm.rerank import Reranker
from rank_llm.rerank.pointwise.jina_reranker import JinaReranker

INLINE_REQUEST = {
    "query": {"text": "how long is life cycle of flea", "qid": "264014"},
    "candidates": [
        {
            "doc": {
                "contents": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle."
            },
            "docid": "4834547",
            "score": 14.97,
        },
        {
            "doc": {
                "contents": "A flea can live up to a year, but its general lifespan depends on its living conditions, such as the availability of hosts."
            },
            "docid": "5611210",
            "score": 15.78,
        },
        {
            "doc": {
                "contents": "Basketball is one of the most popular sports in the United States, with millions of fans worldwide."
            },
            "docid": "0000001",
            "score": 1.00,
        },
        {
            "doc": {
                "contents": "The flea larvae spin cocoons around themselves in which they move to the last phase of the flea life cycle and become adult fleas."
            },
            "docid": "96852",
            "score": 14.22,
        },
        {
            "doc": {
                "contents": "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage."
            },
            "docid": "0000002",
            "score": 0.50,
        },
        {
            "doc": {
                "contents": "The cat flea's primary host is the domestic cat, but it is also the primary flea infesting dogs in most of the world."
            },
            "docid": "4239616",
            "score": 13.95,
        },
    ],
}


def _print_sample(results, max_queries: int = 3, top_k: int = 5) -> None:
    for res in results[:max_queries]:
        print(f"\nqid={res.query.qid}  query={res.query.text!r}")
        for rank, c in enumerate(res.candidates[:top_k], 1):
            snippet = (
                c.doc.get("contents") or c.doc.get("text") or c.doc.get("segment", "")
            )
            print(f"  [{rank}] docid={c.docid}  score={c.score:.4f}  {snippet[:80]}...")


def _run_eval(results, qrels: str) -> None:
    from rank_llm.evaluation.trec_eval import EvalFunction

    metrics = [
        ("nDCG@10", ["-c", "-m", "ndcg_cut.10"]),
        ("MAP@100", ["-c", "-m", "map_cut.100", "-l2"]),
        ("Recall@20", ["-c", "-m", "recall.20"]),
        ("Recall@100", ["-c", "-m", "recall.100"]),
    ]
    for label, eval_args in metrics:
        value = EvalFunction.from_results(results, qrels, eval_args)
        print(f"  {label:12s} {value}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rerank with Jina Reranker v3 (local HuggingFace model)."
    )
    p.add_argument(
        "--model",
        default="jinaai/jina-reranker-v3",
        help="HuggingFace model ID (default: jinaai/jina-reranker-v3).",
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--max-passage-words",
        type=int,
        default=None,
        help="Truncate each passage to this many words. "
        "When unset, derived from context_size / num_docs.",
    )
    p.add_argument("--context-size", type=int, default=131_072)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="auto")
    p.add_argument(
        "--inline",
        action="store_true",
        help="Use inline candidates instead of retrieving from an index "
        "(no pyserini dependency required).",
    )
    args = p.parse_args()

    # --- Load candidates ---
    if args.inline:
        requests = [from_dict(data_class=Request, data=INLINE_REQUEST)]
        print(
            f"Using inline request: 1 query, {len(requests[0].candidates)} candidates."
        )
        qrels = None
    else:
        from rank_llm.retrieve import TOPICS
        from rank_llm.retrieve.retriever import Retriever

        requests = Retriever.from_dataset_with_prebuilt_index(args.dataset, k=args.k)
        print(f"Retrieved {len(requests)} queries from {args.dataset} (k={args.k}).")
        qrels = TOPICS.get(args.dataset)

    # --- Retrieval-only evaluation ---
    if qrels:
        print(f"\n{'=' * 60}")
        print(f"Retrieval metrics  (BM25, k={args.k})")
        print(f"{'=' * 60}")
        _run_eval(requests, qrels)

    # --- Build reranker ---
    print(f"\nLoading Jina model: {args.model} (device={args.device}) ...")
    coordinator = JinaReranker(
        model=args.model,
        context_size=args.context_size,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
        max_passage_words=args.max_passage_words,
    )
    reranker = Reranker(coordinator)
    kwargs = {"populate_invocations_history": True}

    # --- Rerank ---
    print("\n--- Reranking ---")
    results = reranker.rerank_batch(requests, **kwargs)
    print(f"Reranked {len(results)} queries.")
    _print_sample(results)

    # --- Reranking evaluation ---
    if qrels:
        print(f"\n{'=' * 60}")
        print(f"Reranking metrics  ({args.model})")
        print(f"{'=' * 60}")
        _run_eval(results, qrels)

    # --- Write outputs ---
    Path("demo_outputs/").mkdir(parents=True, exist_ok=True)
    writer = DataWriter(results)
    writer.write_in_jsonl_format("demo_outputs/rerank_results_jina.jsonl")
    writer.write_in_trec_eval_format("demo_outputs/rerank_results_jina.txt")
    writer.write_inference_invocations_history(
        "demo_outputs/inference_invocations_history_jina.json"
    )
    print("\nOutputs written to demo_outputs/rerank_results_jina.*")


if __name__ == "__main__":
    main()

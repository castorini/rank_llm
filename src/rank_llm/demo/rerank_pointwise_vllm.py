"""
Demo: generative pointwise reranking (yes/no + logprobs) against a local
OpenAI-compatible vLLM server, using the same dl19 retrieval path as
``rerank_monot5.py`` / ``rerank_qwen.py`` (BM25 + prebuilt index).

Prerequisites:
  1. pip install -e '.[pyserini,vllm]'  (retrieval needs pyserini / JDK; vLLM
     needs openai + transformers at minimum).
  2. Start vLLM with an OpenAI-compatible HTTP API, for example (Qwen reranker
     model, long context, prompt token details for usage logging):

       RANK_MODEL_ID="Qwen/Qwen3-Reranker-0.6B"
       RANK_PORT=8765
       RANK_VLLM_LOG=/tmp/vllm_rerank.log   # or any writable path
       CUDA_VISIBLE_DEVICES=0 vllm serve "$RANK_MODEL_ID" \\
         --port "$RANK_PORT" \\
         --dtype auto \\
         --gpu-memory-utilization 0.9 \\
         --enable-prompt-tokens-details \\
         --enable-prefix-caching \\
         --max-model-len 32768 \\
         >> "$RANK_VLLM_LOG" 2>&1 &

     PointwiseVLLM talks to ``http://127.0.0.1:${RANK_PORT}/v1`` by default.

Usage (from repo root, after indexes are available):

  python src/rank_llm/demo/rerank_pointwise_vllm.py \\
      --base-url http://127.0.0.1:8765/v1 \\
      --model Qwen/Qwen3-Reranker-0.6B
"""

from __future__ import annotations

import argparse
import asyncio
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
from rank_llm.rerank.pointwise.pointwise_vllm import PointwiseVLLM
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
    p = argparse.ArgumentParser()
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
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8765/v1"),
        help="OpenAI-compatible base URL (must end with /v1; match vLLM --port).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Reranker-0.6B"),
        help="Model id as registered on the server (default: Qwen reranker above).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-concurrent", type=int, default=None)
    p.add_argument(
        "--max-passage-words",
        type=int,
        default=None,
        help="Per-passage word truncation limit (default: None = token-only truncation).",
    )
    p.add_argument(
        "--async-sample-queries",
        type=int,
        default=2,
        help="Number of dl19 queries to rerun with rerank_batch_async (default: 2). "
        "Set to 0 to skip the async demo.",
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

    # --- Retrieval-only evaluation ---
    if qrels and run_eval:
        print(f"\n{'=' * 60}")
        print(f"Retrieval metrics  (BM25, k={args.k})")
        print(f"{'=' * 60}")
        _print_eval(requests, qrels)

    # --- Rerank ---

    coordinator = PointwiseVLLM(
        model=args.model,
        base_url=args.base_url,
        batch_size=args.batch_size,
        max_concurrent_llm_calls=args.max_concurrent,
        max_passage_words=args.max_passage_words,
    )
    reranker = Reranker(coordinator)
    kwargs = {"populate_invocations_history": True}

    print("\n--- Sync rerank_batch (full dataset) ---")
    sync_out = reranker.rerank_batch(requests, **kwargs)
    print(f"Sync: {len(sync_out)} results.")
    _print_sample(sync_out)

    # --- Reranking evaluation ---
    if qrels and run_eval:
        print(f"\n{'=' * 60}")
        print(f"Reranking metrics  ({args.model})")
        print(f"{'=' * 60}")
        _print_eval(sync_out, qrels)

    if not args.skip_eval:

        async def async_demo():
            if args.async_sample_queries <= 0:
                return
            sample = requests[: args.async_sample_queries]
            print(
                f"\n--- Async rerank_batch_async (first {len(sample)} queries only) ---"
            )
            async_out = await reranker.rerank_batch_async(sample, **kwargs)
            print(f"Async sample: {len(async_out)} results.")
            _print_sample(async_out)

        asyncio.run(async_demo())

    # --- Save outputs ---
    model_tag = args.model.split("/")[-1].lower()
    if args.output_dir:
        out_path = Path(args.output_dir) / model_tag / args.dataset
    else:
        out_path = Path("demo_outputs")
    out_path.mkdir(parents=True, exist_ok=True)

    writer = DataWriter(sync_out)
    writer.write_in_jsonl_format(str(out_path / "rerank.jsonl"))
    writer.write_in_trec_eval_format(str(out_path / "rerank.txt"))
    writer.write_inference_invocations_history(str(out_path / "invocations.json"))


if __name__ == "__main__":
    main()

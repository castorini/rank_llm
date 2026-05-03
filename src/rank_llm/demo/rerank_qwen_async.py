"""
Concurrent async listwise reranking demo against a local OpenAI-compatible
vLLM server (e.g. Qwen3.5-9B).

Uses multiple ``await reranker.rerank_async(...)`` calls on one event loop and
one ``Reranker`` instance so sliding-window LLM work overlaps across queries.

The reranker is constructed **before** ``asyncio.run`` (sync setup); only the
gather phase runs under the event loop, matching the one-loop contract and
avoiding nested ``asyncio.run`` during model init.

Prerequisites:
  1. pip install -e '.[pyserini,vllm]'
  2. Start vLLM with an OpenAI-compatible HTTP API, for example:

       MODEL_ID="Qwen/Qwen3.5-9B"
       PORT=8765
       CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_ID" \\
         --port "$PORT" \\
         --dtype auto \\
         --gpu-memory-utilization 0.9 \\
         --enable-prompt-tokens-details \\
         --enable-prefix-caching \\
         --max-model-len 32768 \\
         >> /tmp/vllm_rerank.log 2>&1 &

Usage (from repo root, after indexes are available):

  python src/rank_llm/demo/rerank_qwen_async.py \\
      --base-url http://127.0.0.1:8765/v1 \\
      --model Qwen/Qwen3.5-9B
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

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import DataWriter, Request, Result
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM
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


async def rerank_requests_concurrently(
    reranker: Reranker,
    requests: list[Request],
    **kwargs,
) -> list[Result]:
    return list(
        await asyncio.gather(
            *(reranker.rerank_async(req, **kwargs) for req in requests)
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Async listwise reranking demo with a vLLM server."
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
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8765/v1"),
        help="OpenAI-compatible base URL (must end with /v1; match vLLM --port).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B"),
        help="Model id as registered on the server (default: Qwen/Qwen3.5-9B).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context size for the model (default: 4096).",
    )
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument(
        "--max-passage-words",
        type=int,
        default=300,
        help="Per-passage word truncation limit (default: 300).",
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

    coordinator = RankListwiseOSLLM(
        model=args.model,
        context_size=args.context_size,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        base_url=args.base_url,
        max_passage_words=args.max_passage_words,
    )
    reranker = Reranker(coordinator)
    kwargs = {"populate_invocations_history": True}

    print(f"\n--- Async concurrent reranking ({len(requests)} queries) ---")
    rerank_results = asyncio.run(
        rerank_requests_concurrently(reranker, requests, **kwargs)
    )
    print(f"Reranked {len(rerank_results)} results.")
    _print_sample(rerank_results)

    analyzer = ResponseAnalyzer.from_inline_results(rerank_results, use_alpha=False)
    error_counts = analyzer.count_errors()
    print(f"\nResponse analysis: {error_counts!r}")

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

"""
Demo: Qwen3 pointwise reranking against a local OpenAI-compatible vLLM server.

Prerequisites:
  1. Install extras: pip install -e '.[vllm]'  (or at least openai + transformers).
  2. Serve a Qwen3 chat model with vLLM, e.g.:
       vllm serve Qwen/Qwen3-0.6B --port 8000
     (Adjust model id and port; pass the same to this script.)

Usage:
  .venv/bin/python src/rank_llm/demo/rerank_qwen3_pointwise_vllm.py \\
      --base-url http://127.0.0.1:8000/v1 \\
      --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
repo_src = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank import Reranker
from rank_llm.rerank.pointwise.qwen3_pointwise_vllm import Qwen3PointwiseVLLM


def _demo_requests() -> list[Request]:
    return [
        Request(
            query=Query(text="capital of France", qid="demo1"),
            candidates=[
                Candidate(
                    docid="1",
                    score=0.5,
                    doc={"contents": "Paris is the capital and largest city of France."},
                ),
                Candidate(
                    docid="2",
                    score=0.6,
                    doc={"contents": "Berlin is known for museums and nightlife."},
                ),
                Candidate(
                    docid="3",
                    score=0.4,
                    doc={"contents": "Lyon is a city in France."},
                ),
            ],
        )
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL (vLLM default ends with /v1).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", None),
        help="Model id as registered on the server (default: first from /v1/models).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-concurrent", type=int, default=None)
    args = p.parse_args()

    coordinator = Qwen3PointwiseVLLM(
        model=args.model or "placeholder",
        base_url=args.base_url,
        batch_size=args.batch_size,
        max_concurrent_llm_calls=args.max_concurrent,
    )
    reranker = Reranker(coordinator)
    requests = _demo_requests()

    print("--- Sync rerank_batch ---")
    sync_out = reranker.rerank_batch(requests, rank_end=10)
    for res in sync_out:
        print(f"qid={res.query.qid} text={res.query.text!r}")
        for c in res.candidates:
            print(f"  docid={c.docid} score={c.score:.4f}")

    async def async_demo():
        print("--- Async rerank_batch_async ---")
        async_out = await reranker.rerank_batch_async(requests, rank_end=10)
        for res in async_out:
            print(f"qid={res.query.qid} text={res.query.text!r}")
            for c in res.candidates:
                print(f"  docid={c.docid} score={c.score:.4f}")

    asyncio.run(async_demo())


if __name__ == "__main__":
    main()

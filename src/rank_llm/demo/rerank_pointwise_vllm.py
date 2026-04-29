"""
Demo: generative pointwise reranking (yes/no + logprobs) against a local
OpenAI-compatible vLLM server.

Prerequisites:
  1. pip install -e '.[vllm]'  (or at least openai + transformers).
  2. Serve a chat model with vLLM, e.g.:
       vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000

Usage:
  python src/rank_llm/demo/rerank_pointwise_vllm.py \\
      --base-url http://127.0.0.1:8000/v1 \\
      --model meta-llama/Llama-3.2-1B-Instruct
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
from rank_llm.rerank.pointwise.pointwise_vllm import PointwiseVLLM


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

    coordinator = PointwiseVLLM(
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

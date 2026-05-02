"""
Demo: API-based reranking with Jina Reranker v3, using the same
dl19 retrieval path as ``rerank_jina.py`` (BM25 + prebuilt index).

Instead of running the model locally, this script calls the Jina
Reranker REST API (https://api.jina.ai/v1/rerank).  It respects the
free-tier rate limits (500 RPM, 1M TPM, concurrency 5) by throttling
requests and checking rate-limit headers.

Get your Jina AI API key for free: https://jina.ai/?sui=apikey

Prerequisites:
  pip install -e '.[pyserini]'
  # Set JINA_API_KEY in your environment or in .env.local

Usage (from repo root, after indexes are available):

  python src/rank_llm/demo/rerank_jina_api.py

  python src/rank_llm/demo/rerank_jina_api.py \\
      --model jina-reranker-v3 \\
      --window-size 32 \\
      --max-passage-words 512
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import requests as http_requests
from ftfy import fix_text
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, InferenceInvocation, Result
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.retrieve import TOPICS
from rank_llm.retrieve.retriever import Retriever

logger = logging.getLogger(__name__)

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
FREE_TIER_RPM = 500
MIN_REQUEST_INTERVAL = 60.0 / FREE_TIER_RPM
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0


def _load_api_key() -> str:
    key = os.environ.get("JINA_API_KEY")
    if key:
        return key
    env_local = Path(__file__).resolve().parents[3] / ".env.local"
    if env_local.exists():
        for line in env_local.read_text().splitlines():
            line = line.strip()
            if line.startswith("JINA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "JINA_API_KEY not found. Set it as an environment variable or in .env.local. "
        "Get your free key at https://jina.ai/?sui=apikey"
    )


def _extract_doc_text(doc: dict[str, Any], max_words: int | None = None) -> str:
    if "text" in doc:
        content = doc["text"]
    elif "segment" in doc:
        content = doc["segment"]
    elif "contents" in doc:
        content = doc["contents"]
    elif "content" in doc:
        content = doc["content"]
    elif "body" in doc:
        content = doc["body"]
    elif "passage" in doc:
        content = doc["passage"]
    else:
        content = str(doc)

    if "title" in doc and doc["title"]:
        content = "Title: " + doc["title"] + " Content: " + content

    content = fix_text(content.strip())

    if max_words is not None:
        content = " ".join(content.split()[:max_words])

    return content


def _call_jina_rerank(
    api_key: str,
    model: str,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[dict]:
    """Call the Jina Reranker API with retry + rate-limit back-off."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = http_requests.post(
                JINA_RERANK_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
        except http_requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF_BASE**attempt
            logger.warning(
                "Network error (attempt %d/%d): %s – retrying in %.1fs",
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            retry_after = float(
                resp.headers.get("Retry-After", RETRY_BACKOFF_BASE**attempt)
            )
            logger.warning(
                "Rate limited (429) – sleeping %.1fs (attempt %d/%d)",
                retry_after,
                attempt,
                MAX_RETRIES,
            )
            time.sleep(retry_after)
            continue

        if resp.status_code >= 500:
            wait = RETRY_BACKOFF_BASE**attempt
            logger.warning(
                "Server error %d (attempt %d/%d) – retrying in %.1fs",
                resp.status_code,
                attempt,
                MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
            continue

        resp.raise_for_status()
        data = resp.json()
        return data["results"]

    resp.raise_for_status()
    return []


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


def _candidate_comparator(x, y) -> int:
    if x.score < y.score:
        return -1
    elif x.score > y.score:
        return 1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rerank with Jina Reranker v3 via REST API."
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
        default="jina-reranker-v3",
        help="Jina reranker model name (default: jina-reranker-v3).",
    )
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument(
        "--max-passage-words",
        type=int,
        default=512,
        help="Truncate each passage to this many words (default: 512).",
    )
    args = p.parse_args()

    api_key = _load_api_key()
    print(f"Jina API key loaded (ending ...{api_key[-4:]})")

    requests_data = Retriever.from_dataset_with_prebuilt_index(args.dataset, k=args.k)
    print(f"Loaded {len(requests_data)} requests from {args.dataset} (k={args.k}).")

    qrels = TOPICS.get(args.dataset)

    if qrels:
        print(f"\n{'=' * 60}")
        print(f"Retrieval metrics  (BM25, k={args.k})")
        print(f"{'=' * 60}")
        _print_eval(requests_data, qrels)

    rerank_results = [
        Result(
            query=copy.deepcopy(req.query),
            candidates=copy.deepcopy(req.candidates),
            invocations_history=[],
        )
        for req in requests_data
    ]

    total_api_calls = sum(
        (len(r.candidates) + args.window_size - 1) // args.window_size
        for r in rerank_results
    )
    print(f"\n--- Reranking via API ({args.model}) ---")
    print(f"  window_size={args.window_size}, top_n={args.window_size}")
    print(f"  Estimated API calls: {total_api_calls} (free tier: {FREE_TIER_RPM} RPM)")

    last_request_time = 0.0

    with tqdm(total=total_api_calls, desc="Jina API reranking") as pbar:
        for result in rerank_results:
            query_text = result.query.text
            candidates = result.candidates
            num_candidates = len(candidates)
            doc_texts = [
                _extract_doc_text(c.doc, max_words=args.max_passage_words)
                for c in candidates
            ]

            all_scores: list[float] = [0.0] * num_candidates

            for chunk_start in range(0, num_candidates, args.window_size):
                chunk_end = min(chunk_start + args.window_size, num_candidates)
                chunk_docs = doc_texts[chunk_start:chunk_end]

                elapsed = time.monotonic() - last_request_time
                if elapsed < MIN_REQUEST_INTERVAL:
                    time.sleep(MIN_REQUEST_INTERVAL - elapsed)

                api_results = _call_jina_rerank(
                    api_key=api_key,
                    model=args.model,
                    query=query_text,
                    documents=chunk_docs,
                    top_n=len(chunk_docs),
                )
                last_request_time = time.monotonic()

                for item in api_results:
                    all_scores[chunk_start + item["index"]] = float(
                        item["relevance_score"]
                    )

                result.invocations_history.append(
                    InferenceInvocation(
                        prompt=(
                            f"query={query_text!r}, "
                            f"docs[{chunk_start}:{chunk_end}] "
                            f"({len(chunk_docs)} docs)"
                        ),
                        response=", ".join(
                            f"[{chunk_start + item['index']}]="
                            f"{item['relevance_score']:.4f}"
                            for item in sorted(api_results, key=lambda x: x["index"])
                        ),
                        input_token_count=0,
                        output_token_count=0,
                    )
                )

                pbar.update(1)

            for idx, score in enumerate(all_scores):
                result.candidates[idx].score = score

            result.candidates.sort(key=cmp_to_key(_candidate_comparator), reverse=True)

    print(f"Reranked {len(rerank_results)} queries.")
    _print_sample(rerank_results)

    if qrels:
        print(f"\n{'=' * 60}")
        print(f"Reranking metrics  ({args.model}, API)")
        print(f"{'=' * 60}")
        _print_eval(rerank_results, qrels)

    writer = DataWriter(rerank_results)
    Path("demo_outputs/").mkdir(parents=True, exist_ok=True)
    writer.write_in_jsonl_format("demo_outputs/rerank_results_jina_api.jsonl")
    writer.write_in_trec_eval_format("demo_outputs/rerank_results_jina_api.txt")
    writer.write_inference_invocations_history(
        "demo_outputs/inference_invocations_history_jina_api.json"
    )


if __name__ == "__main__":
    main()

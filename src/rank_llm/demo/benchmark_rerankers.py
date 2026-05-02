"""
Comprehensive benchmark: evaluate multiple reranker models on DL and BEIR datasets.

Models evaluated:
  Pointwise: Qwen3-Reranker-0.6B/4B/8B (transformers, yes/no logprobs)
             Jina Reranker v3 (local, cross-attention)
  Listwise:  Qwen3.5-0.8B/2B/4B/9B, Qwen3-0.6B/4B/8B (RankListwiseOSLLM + vLLM)

Datasets:
  DL:   dl19, dl20, dl21, dl22, dl23
  BEIR: scifact, dbpedia, nfc, covid, news, signal, robust04

Usage:
  # Smoke test (1 query per dataset per model)
  python src/rank_llm/demo/benchmark_rerankers.py --max-queries 1

  # Full run
  python src/rank_llm/demo/benchmark_rerankers.py

Results are saved under benchmark_outputs/{model_short}/{dataset}/ and never overwrite
each other. A summary JSON is written to benchmark_outputs/summary.json.
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
from functools import cmp_to_key
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

import torch
from ftfy import fix_text
from tqdm import tqdm

from rank_llm.data import DataWriter, Request, Result
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.retrieve.retriever import Retriever
from rank_llm.retrieve.topics_dict import TOPICS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

DL_DATASETS = ["dl19", "dl20"]
BEIR_DATASETS = ["scifact", "dbpedia", "nfc", "covid", "news", "signal", "robust04"]

QWEN3_RERANKER_MODELS = [
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-8B",
]
JINA_MODEL = "jinaai/jina-reranker-v3"
LISTWISE_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

EVAL_METRICS: list[tuple[str, list[str]]] = [
    ("nDCG@10", ["-c", "-m", "ndcg_cut.10"]),
    ("MAP@100", ["-c", "-m", "map_cut.100", "-l2"]),
    ("Recall@20", ["-c", "-m", "recall.20"]),
    ("Recall@100", ["-c", "-m", "recall.100"]),
]

QWEN3_RERANKER_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QWEN3_RERANKER_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def model_short(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def destroy_vllm():
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass
    cleanup_gpu()


def _extract_doc_text(doc: dict[str, Any], max_words: int | None = None) -> str:
    for key in ("text", "segment", "contents", "content", "body", "passage"):
        if key in doc:
            content = doc[key]
            break
    else:
        content = str(doc)
    if "title" in doc and doc["title"]:
        content = "Title: " + doc["title"] + " Content: " + content
    content = fix_text(content.strip())
    if max_words is not None:
        content = " ".join(content.split()[:max_words])
    return content


def _candidate_cmp(x, y) -> int:
    return -1 if x.score < y.score else (1 if x.score > y.score else 0)


def evaluate_results(results: list[Result], qrels: str) -> dict[str, float]:
    metrics = {}
    for label, eval_args in EVAL_METRICS:
        try:
            value = EvalFunction.from_results(results, qrels, eval_args)
            if isinstance(value, str):
                parts = value.strip().split()
                value = float(parts[-1])
            metrics[label] = value
        except Exception as exc:
            logger.warning("Eval %s failed: %s", label, exc)
            metrics[label] = None
    return metrics


def save_run(
    results: list[Result],
    metrics: dict,
    output_dir: Path,
    dataset: str,
):
    ds_dir = output_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    writer = DataWriter(results)
    writer.write_in_jsonl_format(str(ds_dir / "rerank.jsonl"))
    writer.write_in_trec_eval_format(str(ds_dir / "rerank.txt"))
    writer.write_inference_invocations_history(str(ds_dir / "invocations.json"))
    with open(ds_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ─── Step 1: Pre-cache retrieval results ─────────────────────────────────────


def load_retrieval_cache(
    datasets: list[str], k: int, max_queries: int | None
) -> dict[str, list[Request]]:
    cache: dict[str, list[Request]] = {}
    for ds in datasets:
        logger.info("Retrieving %s (k=%d) ...", ds, k)
        try:
            requests = Retriever.from_dataset_with_prebuilt_index(ds, k=k)
            if max_queries is not None:
                requests = requests[:max_queries]
            cache[ds] = requests
            logger.info("  %s: %d queries loaded.", ds, len(requests))
        except Exception as exc:
            logger.error("  Failed to load %s: %s", ds, exc)
    return cache


# ─── Step 2: Qwen3-Reranker pointwise (transformers, yes/no logprobs) ────────


def _qwen3_reranker_score_batch(
    model,
    tokenizer,
    query: str,
    doc_texts: list[str],
    batch_size: int = 16,
    max_length: int = 8192,
) -> list[float]:
    prefix = (
        f"<|im_start|>system\n{QWEN3_RERANKER_SYSTEM}<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")

    pairs = [
        f"<Instruct>: {QWEN3_RERANKER_TASK}\n<Query>: {query}\n<Document>: {doc}"
        for doc in doc_texts
    ]

    all_scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        inputs = tokenizer(
            batch_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = prefix_tokens + ids + suffix_tokens
        inputs = tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        true_vec = logits[:, token_true_id]
        false_vec = logits[:, token_false_id]
        stacked = torch.stack([false_vec, true_vec], dim=1)
        probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = probs[:, 1].exp().tolist()
        all_scores.extend(scores)

    return all_scores


def run_qwen3_reranker(
    model_name: str,
    retrieval_cache: dict[str, list[Request]],
    all_metrics: dict,
    output_base: Path,
    max_passage_words: int = 512,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    short = model_short(model_name)
    out_dir = output_base / short
    logger.info("=" * 70)
    logger.info("Loading Qwen3-Reranker: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        .cuda()
        .eval()
    )

    for ds, requests in retrieval_cache.items():
        logger.info("  [%s] Reranking %s (%d queries) ...", short, ds, len(requests))
        qrels = TOPICS.get(ds)
        results: list[Result] = []
        for req in tqdm(requests, desc=f"{short}/{ds}"):
            result = Result(
                query=copy.deepcopy(req.query),
                candidates=copy.deepcopy(req.candidates),
                invocations_history=[],
            )
            doc_texts = [
                _extract_doc_text(c.doc, max_words=max_passage_words)
                for c in result.candidates
            ]
            scores = _qwen3_reranker_score_batch(
                model, tokenizer, result.query.text, doc_texts
            )
            for idx, score in enumerate(scores):
                result.candidates[idx].score = score
            result.candidates.sort(key=cmp_to_key(_candidate_cmp), reverse=True)
            results.append(result)

        metrics = evaluate_results(results, qrels) if qrels else {}
        save_run(results, metrics, out_dir, ds)
        all_metrics.setdefault(short, {})[ds] = metrics
        if qrels:
            logger.info("    nDCG@10 = %s", metrics.get("nDCG@10"))

    del model, tokenizer
    cleanup_gpu()
    logger.info("Cleaned up %s", model_name)


# ─── Step 3: Jina Reranker v3 local ──────────────────────────────────────────


def run_jina_local(
    retrieval_cache: dict[str, list[Request]],
    all_metrics: dict,
    output_base: Path,
):
    from rank_llm.rerank import Reranker
    from rank_llm.rerank.pointwise.jina_reranker import JinaReranker

    short = model_short(JINA_MODEL)
    out_dir = output_base / short
    logger.info("=" * 70)
    logger.info("Loading Jina Reranker v3 (local)")

    coordinator = JinaReranker(
        model=JINA_MODEL,
        context_size=131_072,
        device="cuda",
        window_size=32,
        batch_size=2,
        max_passage_words=512,
    )
    reranker = Reranker(coordinator)

    for ds, requests in retrieval_cache.items():
        logger.info("  [%s] Reranking %s (%d queries) ...", short, ds, len(requests))
        qrels = TOPICS.get(ds)
        results = reranker.rerank_batch(requests, populate_invocations_history=True)
        metrics = evaluate_results(results, qrels) if qrels else {}
        save_run(results, metrics, out_dir, ds)
        all_metrics.setdefault(short, {})[ds] = metrics
        if qrels:
            logger.info("    nDCG@10 = %s", metrics.get("nDCG@10"))

    del reranker, coordinator
    cleanup_gpu()
    logger.info("Cleaned up Jina Reranker v3")


# ─── Step 4: Listwise reranking (RankListwiseOSLLM + vLLM) ───────────────────


def run_listwise(
    model_name: str,
    retrieval_cache: dict[str, list[Request]],
    all_metrics: dict,
    output_base: Path,
):
    from rank_llm.rerank import Reranker
    from rank_llm.rerank.listwise import RankListwiseOSLLM

    short = model_short(model_name)
    out_dir = output_base / short
    logger.info("=" * 70)
    logger.info("Loading listwise model: %s", model_name)

    coordinator = RankListwiseOSLLM(
        model=model_name,
        context_size=4096,
        window_size=20,
        stride=10,
        batch_size=32,
        num_gpus=1,
    )
    reranker = Reranker(coordinator)

    for ds, requests in retrieval_cache.items():
        logger.info("  [%s] Reranking %s (%d queries) ...", short, ds, len(requests))
        qrels = TOPICS.get(ds)
        try:
            results = reranker.rerank_batch(requests, populate_invocations_history=True)
        except Exception as exc:
            logger.error("    FAILED on %s/%s: %s", short, ds, exc)
            traceback.print_exc()
            all_metrics.setdefault(short, {})[ds] = {"error": str(exc)}
            continue

        metrics = evaluate_results(results, qrels) if qrels else {}
        save_run(results, metrics, out_dir, ds)
        all_metrics.setdefault(short, {})[ds] = metrics
        if qrels:
            logger.info("    nDCG@10 = %s", metrics.get("nDCG@10"))

    del reranker, coordinator
    destroy_vllm()
    logger.info("Cleaned up %s", model_name)


# ─── Summary tables ──────────────────────────────────────────────────────────


def print_table(all_metrics: dict, datasets: list[str], title: str):
    models = [m for m in all_metrics if any(d in all_metrics[m] for d in datasets)]
    if not models:
        print(f"\n{title}: no results.\n")
        return

    metric_key = "nDCG@10"
    header = f"{'Model':<30s}" + "".join(f"{d:>10s}" for d in datasets)
    print(f"\n{'=' * len(header)}")
    print(f"{title}")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for m in models:
        row = f"{m:<30s}"
        for d in datasets:
            entry = all_metrics.get(m, {}).get(d, {})
            if isinstance(entry, dict) and metric_key in entry:
                val = entry[metric_key]
                row += (
                    f"{val:>10.4f}" if isinstance(val, int | float) else f"{'ERR':>10s}"
                )
            else:
                row += f"{'--':>10s}"
        print(row)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    import argparse

    p = argparse.ArgumentParser(description="Benchmark multiple reranker models.")
    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit queries per dataset (e.g. 1 for smoke test).",
    )
    p.add_argument("--k", type=int, default=100, help="Top-k retrieval (default 100).")
    p.add_argument(
        "--output-dir",
        default="benchmark_outputs",
        help="Output directory (default: benchmark_outputs).",
    )
    args = p.parse_args()
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    all_datasets = DL_DATASETS + BEIR_DATASETS
    all_metrics: dict[str, dict[str, dict]] = {}

    # -- Pre-cache retrieval results (one pass for all models) --
    logger.info("Pre-caching retrieval results for %d datasets ...", len(all_datasets))
    retrieval_cache = load_retrieval_cache(
        all_datasets, k=args.k, max_queries=args.max_queries
    )
    logger.info("Cached %d datasets.", len(retrieval_cache))

    # -- BM25 baseline metrics --
    bm25_key = "BM25-baseline"
    for ds, requests in retrieval_cache.items():
        qrels = TOPICS.get(ds)
        if qrels:
            all_metrics.setdefault(bm25_key, {})[ds] = evaluate_results(requests, qrels)

    # -- Pointwise: Qwen3-Reranker --
    for model_name in QWEN3_RERANKER_MODELS:
        try:
            run_qwen3_reranker(model_name, retrieval_cache, all_metrics, output_base)
        except Exception as exc:
            logger.error("FAILED model %s: %s", model_name, exc)
            traceback.print_exc()
            cleanup_gpu()

    # -- Pointwise: Jina Reranker v3 local --
    try:
        run_jina_local(retrieval_cache, all_metrics, output_base)
    except Exception as exc:
        logger.error("FAILED Jina: %s", exc)
        traceback.print_exc()
        cleanup_gpu()

    # -- Listwise: Qwen3.5 and Qwen3 --
    for model_name in LISTWISE_MODELS:
        try:
            run_listwise(model_name, retrieval_cache, all_metrics, output_base)
        except Exception as exc:
            logger.error("FAILED model %s: %s", model_name, exc)
            traceback.print_exc()
            destroy_vllm()

    # -- Save summary --
    with open(output_base / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Summary saved to %s/summary.json", output_base)

    # -- Print tables --
    print_table(all_metrics, DL_DATASETS, "DL Datasets (nDCG@10)")
    print_table(all_metrics, BEIR_DATASETS, "BEIR Datasets (nDCG@10)")


if __name__ == "__main__":
    main()

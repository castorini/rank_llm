"""
Listwise reranking demo with RankGPT (OpenAI chat models such as
``gpt-4o-mini``) on top of BM25 first-stage retrieval from a Pyserini
prebuilt index.

RankGPT is the original sliding-window listwise prompting approach: the
prompt instructs the model to output a permutation of the candidate list
inside a window, and the window slides backwards across the full candidate
set with a configurable overlap. Per-query cost scales with the number of
windows and the model's price-per-token, so smaller k and tighter strides
keep the demo cheap.

Prerequisites:
  pip install -e '.[pyserini]'

The OpenAI API key is loaded via ``rank_llm.rerank.get_openai_api_key``
(reads ``OPENAI_API_KEY`` / ``OPEN_AI_API_KEY`` env vars or a ``.env.local``
file at the repo root).

Usage (from repo root):
  python src/rank_llm/demo/rerank_rank_gpt.py
  python src/rank_llm/demo/rerank_rank_gpt.py \\
      --dataset dl20 \\
      --model gpt-4o-mini \\
      --num-queries 5 \\
      --skip-eval
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, Result
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker, get_openai_api_key
from rank_llm.rerank.listwise import SafeOpenai
from rank_llm.retrieve import TOPICS, Retriever

TEMPLATES = files("rank_llm.rerank.prompt_templates")

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
        description="Listwise RankGPT reranking on BM25 first-stage results."
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
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model id (default: gpt-4o-mini).",
    )
    p.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context size budgeted per window (default: 4096).",
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
        "--batch-size",
        type=int,
        default=32,
        help="Per-step batch size (default: 32).",
    )
    p.add_argument(
        "--prompt-template",
        default=None,
        help="Override prompt template path (default: bundled rank_gpt_template.yaml).",
    )
    p.add_argument(
        "--reasoning-effort",
        default=None,
        help="Reasoning effort for o-series models (e.g. 'low', 'medium', 'high').",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Alternate base URL (e.g. for OpenRouter-style OpenAI-compatible providers).",
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

    prompt_template_path = args.prompt_template or str(
        TEMPLATES / "rank_gpt_template.yaml"
    )
    coordinator = SafeOpenai(
        model=args.model,
        context_size=args.context_size,
        keys=get_openai_api_key(),
        prompt_template_path=prompt_template_path,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        reasoning_effort=args.reasoning_effort,
        base_url=args.base_url,
    )
    reranker = Reranker(coordinator)
    kwargs = {
        "populate_invocations_history": True,
        "top_k_retrieve": args.k,
        "rank_end": args.k,
    }

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
    model_tag = args.model.replace("/", "-").lower()
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

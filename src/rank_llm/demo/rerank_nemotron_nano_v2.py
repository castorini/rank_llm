"""
Listwise reranking demo with NVIDIA-Nemotron-Nano-9B-v2 (default
``nvidia/NVIDIA-Nemotron-Nano-9B-v2``) on top of BM25 first-stage retrieval
from a Pyserini prebuilt index.

Nemotron-Nano-v2 ships with both a thinking and a non-thinking prompt
template. Each mode uses its own YAML in
``rank_llm.rerank.prompt_templates`` and toggles ``is_thinking`` on the
RankListwiseOSLLM coordinator. The ``--variant`` flag selects which mode
to run; ``both`` runs them back-to-back (matching the original hardcoded
demo's behavior).

Prerequisites:
  pip install -e '.[pyserini,vllm]'

Usage (from repo root):
  python src/rank_llm/demo/rerank_nemotron_nano_v2.py
  python src/rank_llm/demo/rerank_nemotron_nano_v2.py \\
      --variant thinking \\
      --dataset dl20 \\
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

# --- set spawn before importing libs that might touch CUDA/vLLM ---
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass  # already set, fine

from rank_llm.data import DataWriter, Result
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM
from rank_llm.retrieve import TOPICS
from rank_llm.retrieve.retriever import Retriever

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


def run_mode(
    args: argparse.Namespace,
    requests: list,
    qrels: str | None,
    thinking: bool,
) -> list[Result]:
    mode_name = "thinking" if thinking else "nothink"
    template_path = (
        args.thinking_template if thinking else args.nothinking_template
    ) or str(
        TEMPLATES
        / (
            "nemotron_thinking_template.yaml"
            if thinking
            else "nemotron_nonthinking_template.yaml"
        )
    )

    print("\n=========================================================")
    print(f"Nemotron Nano v2 — {mode_name} mode")
    print("=========================================================")
    print(f"Template: {template_path}")

    coordinator = RankListwiseOSLLM(
        model=args.model,
        prompt_template_path=template_path,
        context_size=args.context_size,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        is_thinking=thinking,
    )
    reranker = Reranker(coordinator)

    rerank_results = reranker.rerank_batch(
        requests,
        populate_invocations_history=True,
        top_k_retrieve=args.k,
    )
    print(f"Reranked {len(rerank_results)} results.")
    _print_sample(rerank_results)

    if qrels and not args.skip_eval:
        print(f"\nReranking metrics  ({args.model}, {mode_name})")
        _print_eval(rerank_results, qrels)

    # --- Save outputs ---
    model_tag = args.model.split("/")[-1].lower() + f"-{mode_name}"
    if args.output_dir:
        out_path = Path(args.output_dir) / model_tag / args.dataset
    else:
        out_path = Path("demo_outputs") / args.dataset / f"nemotron_nano_v2_{mode_name}"
    out_path.mkdir(parents=True, exist_ok=True)

    writer = DataWriter(rerank_results)
    writer.write_in_jsonl_format(str(out_path / "rerank.jsonl"))
    writer.write_in_trec_eval_format(str(out_path / "rerank.txt"))
    writer.write_inference_invocations_history(str(out_path / "invocations.json"))

    return rerank_results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Listwise NVIDIA-Nemotron-Nano-9B-v2 reranking on BM25 first-stage results."
    )
    p.add_argument(
        "--variant",
        choices=("both", "thinking", "nothink"),
        default="both",
        help="Which Nemotron mode to run (default: both — runs thinking then "
        "non-thinking back-to-back, matching the original hardcoded script).",
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
        default="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        help="HuggingFace model id (default: nvidia/NVIDIA-Nemotron-Nano-9B-v2).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-step batch size (default: 32).",
    )
    p.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context size for the model (default: 4096).",
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
        "--thinking-template",
        default=None,
        help="Override prompt template path for thinking mode (default: bundled nemotron_thinking_template.yaml).",
    )
    p.add_argument(
        "--nothinking-template",
        default=None,
        help="Override prompt template path for non-thinking mode (default: bundled nemotron_nonthinking_template.yaml).",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1).",
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
        print(f"\nRetrieval metrics  (BM25, k={args.k})")
        _print_eval(requests, qrels)

    if args.variant in ("both", "thinking"):
        run_mode(args, requests, qrels, thinking=True)
    if args.variant in ("both", "nothink"):
        run_mode(args, requests, qrels, thinking=False)


if __name__ == "__main__":
    main()

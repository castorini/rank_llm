import argparse
import json
import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

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

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import DataWriter, Result
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM
from rank_llm.retrieve import TOPICS, Retriever

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


def load_sampling_kw_dict(args: argparse.Namespace) -> dict[str, Any]:
    """
    Prefer --sampling-json-file > --sampling-json > SAMPLING_JSON env.
    Returns {} when nothing is configured (RankLLM then omits extras).
    """
    if getattr(args, "sampling_json_file", None):
        raw = Path(args.sampling_json_file).read_text(encoding="utf-8")
        blob = json.loads(raw)
    elif getattr(args, "sampling_json", None) not in (None, ""):
        blob = json.loads(args.sampling_json)
    else:
        env_raw = os.environ.get("SAMPLING_JSON") or ""
        if not env_raw.strip():
            return {}
        blob = json.loads(env_raw)
    if not isinstance(blob, dict):
        raise SystemExit("--sampling-json(|-file): root JSON must be an object.")
    return blob


def main():
    p = argparse.ArgumentParser(
        description="Listwise DeepSeek-R1 Qwen3 distill reranking demo."
    )
    p.add_argument(
        "--dataset",
        default="dl19",
        help="TREC dataset name with prebuilt index (default: dl19).",
    )
    p.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        help="HuggingFace model id (default: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=100,
        help="Top-k passages per query from first-stage retrieval (default: 100).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used by the listwise inference handler (default: 32).",
    )
    p.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context size for the model (default: 4096).",
    )
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
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1).",
    )
    p.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking mode for reasoning models (--thinking / --no-thinking; default: enabled).",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=30000,
        help="Max reasoning tokens when --thinking is enabled (default: 30000).",
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
    p.add_argument(
        "--sampling-json-file",
        default=None,
        help=(
            "JSON object file of inference sampling knobs (temperature, "
            "top_k, repetition_penalty, presence_penalty, frequency_penalty, "
            "stop, ...)."
        ),
    )
    p.add_argument(
        "--sampling-json",
        default=None,
        help="Same fields as sampling-json-file, as an inline JSON object string.",
    )
    p.add_argument(
        "--prompt-template-path",
        default=None,
        help=(
            "Path to a custom prompt-template YAML file. Defaults to the packaged "
            "Qwen thinking template when thinking is enabled."
        ),
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

    TEMPLATES = files("rank_llm.rerank.prompt_templates")
    prompt_template_path = args.prompt_template_path

    if prompt_template_path is None and args.thinking:
        prompt_template_path = TEMPLATES / "qwen_thinking_template.yaml"

    coordinator = RankListwiseOSLLM(
        model=args.model,
        context_size=args.context_size,
        prompt_template_path=prompt_template_path,
        num_gpus=args.num_gpus,
        window_size=args.window_size,
        stride=args.stride,
        is_thinking=args.thinking,
        reasoning_token_budget=args.thinking_budget,
        sampling_kwargs=load_sampling_kw_dict(args) or None,
        batch_size=args.batch_size,
        max_passage_words=args.max_passage_words,
    )
    reranker = Reranker(coordinator)
    kwargs = {
        "populate_invocations_history": True,
        "top_k_retrieve": args.k,
        "rank_end": args.k,
    }
    rerank_results = reranker.rerank_batch(requests, **kwargs)
    _print_sample(rerank_results)

    analyzer = ResponseAnalyzer.from_inline_results(rerank_results, use_alpha=False)
    error_counts = analyzer.count_errors()
    print(error_counts.__repr__())

    if qrels and run_eval:
        print(f"\n{'=' * 60}")
        print(f"Reranking metrics  ({args.model})")
        print(f"{'=' * 60}")
        _print_eval(rerank_results, qrels)

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

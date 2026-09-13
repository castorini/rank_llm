"""
BRIGHT reranking demo using OpenAI-compatible vLLM (gpt-oss-20b) with **async** API.

Same inputs and outputs as ``rerank_oss_bright.py``, but requests are reranked
concurrently with ``await reranker.rerank_async(...)``.

Start the default model server before running this script:

```bash
RANK_MODEL_ID="openai/gpt-oss-20b"
RANK_PORT=48003
RANK_VLLM_LOG="vllm_server_48003.log"
CUDA_VISIBLE_DEVICES=0 vllm serve "$RANK_MODEL_ID" \
 --port "$RANK_PORT" \
 --dtype auto \
 --gpu-memory-utilization 0.9 \
 --max-model-len 32768 \
 --enable-prompt-tokens-details \
 --enable-prefix-caching \
 > "$RANK_VLLM_LOG" 2>&1 &
```
"""

import argparse
import asyncio
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

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import DataWriter, Request, Result, read_requests_from_file
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM

BRIGHT_DATASETS = (
    "aops",
    "biology",
    "earth-science",
    "economics",
    "leetcode",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable-living",
    "theoremqa-questions",
    "theoremqa-theorems",
)

FUSIONS = (
    "bm25-splade",
    "bge-splade",
    "bm25-bge",
)

NDCG_10_ARGS = ["-c", "-m", "ndcg_cut.10"]


def _print_sample(results: list[Result], max_queries: int = 2, top_k: int = 5) -> None:
    for res in results[:max_queries]:
        print(f"qid={res.query.qid} text={res.query.text!r}")
        for c in res.candidates[:top_k]:
            print(f"  docid={c.docid} score={c.score:.4f}")


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


async def rerank_requests_async(
    reranker: Reranker,
    requests: list[Request],
    **kwargs: Any,
) -> list[Result]:
    """Rerank all requests concurrently on one event loop and reranker instance."""
    return list(
        await asyncio.gather(
            *(reranker.rerank_async(request, **kwargs) for request in requests)
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Async listwise gpt-oss reranking of stored BRIGHT fusion results."
    )
    p.add_argument(
        "--dataset",
        choices=BRIGHT_DATASETS,
        default="aops",
        help="BRIGHT dataset category to rerank (default: aops).",
    )
    p.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Model id served by vLLM (default: openai/gpt-oss-20b).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=100,
        help="Top-k stored fusion results to load and rerank (default: 100).",
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
        help=(
            "Output directory. When set, outputs go to "
            "{output_dir}/{model_tag}/{dataset}/{fusion}/."
        ),
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip inline evaluation (useful in batch benchmark runs).",
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
        default=16384,
        help="Context size for the model (default: 16384).",
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
        "--max-passage-words",
        type=int,
        default=300,
        help="Per-passage word truncation limit (default: 300).",
    )
    p.add_argument(
        "--sampling-json",
        default=None,
        help="Inference sampling knobs as an inline JSON object string.",
    )
    p.add_argument(
        "--sampling-json-file",
        default=None,
        help="JSON object file of inference sampling knobs.",
    )
    p.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking mode (--thinking / --no-thinking; default: enabled).",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=16384,
        help="Max reasoning tokens when --thinking is enabled (default: 16384).",
    )
    p.add_argument(
        "--fusion",
        choices=FUSIONS,
        default="bm25-splade",
        help="Stored first-stage fusion method (default: bm25-splade).",
    )
    p.add_argument(
        "--bright-dir",
        default="../bright_fusion",
        help="Directory containing BRIGHT retrieve_results and qrels subdirectories.",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:48003/v1",
        help="Base URL of the OpenAI-compatible vLLM server.",
    )
    p.add_argument(
        "--prompt-template-path",
        default=None,
        help=(
            "Path to a custom prompt-template YAML file. Defaults to the packaged "
            "RankZephyr template."
        ),
    )
    p.add_argument(
        "--use-alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use alphabetic passage identifiers (--use-alpha / --no-use-alpha; default: enabled).",
    )
    args = p.parse_args()

    bright_dir = Path(args.bright_dir)
    file_name = (
        bright_dir
        / "retrieve_results"
        / f"retrieve_results_bright-{args.dataset}-rrf-{args.fusion}_top{args.k}.jsonl"
    )
    retrieve_results = read_requests_from_file(str(file_name))
    if args.num_queries is not None:
        retrieve_results = retrieve_results[: args.num_queries]
    print(
        f"Loaded {len(retrieve_results)} requests from BRIGHT {args.dataset} "
        f"({args.fusion}, k={args.k})."
    )

    qrels = bright_dir / "qrels" / f"qrels.bright-{args.dataset}.txt"
    run_eval = not args.skip_eval

    retrieve_ndcg_10 = None
    if run_eval:
        retrieve_ndcg_10 = EvalFunction.from_results(
            retrieve_results, str(qrels), NDCG_10_ARGS
        )
        print(f"Retrieval nDCG@10: {retrieve_ndcg_10}")

    TEMPLATES = files("rank_llm.rerank.prompt_templates")
    prompt_template_path = args.prompt_template_path
    if prompt_template_path is None:
        prompt_template_path = TEMPLATES / "rank_zephyr_template.yaml"

    coordinator = RankListwiseOSLLM(
        model=args.model,
        context_size=args.context_size,
        prompt_template_path=prompt_template_path,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        max_passage_words=args.max_passage_words,
        is_thinking=args.thinking,
        reasoning_token_budget=args.thinking_budget,
        sampling_kwargs=load_sampling_kw_dict(args) or None,
        use_alpha=args.use_alpha,
        base_url=args.base_url,
    )
    reranker = Reranker(coordinator)
    kwargs = {
        "populate_invocations_history": True,
        "top_k_retrieve": args.k,
        "rank_end": args.k,
    }

    print(f"Reranking {len(retrieve_results)} requests concurrently.")
    rerank_results = asyncio.run(
        rerank_requests_async(reranker, retrieve_results, **kwargs)
    )
    _print_sample(rerank_results)

    analyzer = ResponseAnalyzer.from_inline_results(
        rerank_results, use_alpha=args.use_alpha
    )
    error_counts = analyzer.count_errors()
    print(error_counts.__repr__())

    rerank_ndcg_10 = None
    if run_eval:
        rerank_ndcg_10 = EvalFunction.from_results(
            rerank_results, str(qrels), NDCG_10_ARGS
        )
        print(f"Reranking nDCG@10 ({args.model}): {rerank_ndcg_10}")

    model_tag = args.model.split("/")[-1].lower()
    if args.thinking:
        model_tag += "-thinking"
    else:
        model_tag += "-non-thinking"
    model_tag += "-async"

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = bright_dir / "rerank_results"
    out_path = output_root / model_tag / args.dataset / args.fusion
    out_path.mkdir(parents=True, exist_ok=True)

    writer = DataWriter(rerank_results)
    writer.write_in_jsonl_format(str(out_path / "rerank.jsonl"))
    writer.write_in_trec_eval_format(str(out_path / "rerank.txt"))
    writer.write_inference_invocations_history(str(out_path / "invocations.json"))

    if run_eval:
        with (out_path / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"retrieve": retrieve_ndcg_10, "rerank": rerank_ndcg_10}, f)


if __name__ == "__main__":
    main()

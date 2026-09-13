import argparse
import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import DataWriter, Result
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker, get_openrouter_api_key
from rank_llm.rerank.listwise import SafeOpenai
from rank_llm.retrieve import TOPICS, Retriever

EVAL_METRICS: list[tuple[str, list[str]]] = [
    ("nDCG@10", ["-c", "-m", "ndcg_cut.10"]),
    ("MAP@100", ["-c", "-m", "map_cut.100", "-l2"]),
    ("Recall@20", ["-c", "-m", "recall.20"]),
    ("Recall@100", ["-c", "-m", "recall.100"]),
]


def _print_sample(results: list[Result], max_queries: int = 2, top_k: int = 5) -> None:
    for result in results[:max_queries]:
        print(f"qid={result.query.qid} text={result.query.text!r}")
        for candidate in result.candidates[:top_k]:
            print(f"  docid={candidate.docid} score={candidate.score:.4f}")


def _print_eval(results: list[Result], qrels: str) -> None:
    for label, eval_args in EVAL_METRICS:
        value = EvalFunction.from_results(results, qrels, eval_args)
        print(f"  {label:12s} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listwise minimax-m2 reranking through the OpenRouter API."
    )
    parser.add_argument(
        "--dataset",
        default="dl19",
        help="TREC dataset name with prebuilt index (default: dl19).",
    )
    parser.add_argument(
        "--model",
        default="minimax/minimax-m2:free",
        help="OpenRouter model id (default: minimax/minimax-m2:free).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Top-k passages per query from first-stage retrieval (default: 100).",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Cap the number of queries for a quick smoke test (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used by the listwise inference handler (default: 32).",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context size for the model (default: 4096).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Sliding-window size over candidates (default: 20).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sliding-window stride (default: 10).",
    )
    parser.add_argument(
        "--max-passage-words",
        type=int,
        default=300,
        help="Per-passage word truncation limit (default: 300).",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1/",
        help="OpenAI-compatible OpenRouter base URL.",
    )
    parser.add_argument(
        "--prompt-template-path",
        default=None,
        help=(
            "Path to a custom prompt-template YAML file. Defaults to the packaged "
            "RankZephyr template."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional reasoning effort passed to models that support it.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. When set, outputs go to "
        "{output_dir}/{model_tag}/{dataset}/.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip inline evaluation (useful in batch benchmark runs).",
    )
    args = parser.parse_args()

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

    prompt_template_path = args.prompt_template_path

    TEMPLATES = files("rank_llm.rerank.prompt_templates")
    if prompt_template_path is None:
        prompt_template_path = TEMPLATES / "rank_zephyr_template.yaml"

    coordinator = SafeOpenai(
        model=args.model,
        context_size=args.context_size,
        prompt_template_path=prompt_template_path,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        keys=get_openrouter_api_key(),
        base_url=args.base_url,
        api_type="openai",
        reasoning_effort=args.reasoning_effort,
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

    model_tag = args.model.split("/")[-1].replace(":", "-").lower()
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

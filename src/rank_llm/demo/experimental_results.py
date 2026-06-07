import argparse
import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

TEMPLATES = files("rank_llm.rerank.prompt_templates")
DEFAULT_RERANKERS = [
    "monot5",
    "duot5",
    "lit5",
    "rv",
    "rz",
    "mistral",
    "qwen",
    "llama",
    "gemini",
    "rank_gpt",
    "rank_gpt_apeer",
    "lrl",
]
DEFAULT_ANALYSIS_RERANKERS = [
    "lit5",
    "rv",
    "rz",
    "mistral",
    "qwen",
    "llama",
    "gemini",
    "rank_gpt",
    "rank_gpt_apeer",
    "lrl",
]
DEFAULT_DATASETS = ["dl19", "dl20", "dl21", "dl22", "dl23"]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def create_reranker(name: str, args: argparse.Namespace):
    from rank_llm.rerank import Reranker, get_genai_api_key, get_openai_api_key

    if name == "monot5":
        from rank_llm.rerank.pointwise.monot5 import MonoT5

        return Reranker(MonoT5(args.monot5_model))
    if name == "duot5":
        from rank_llm.rerank.pairwise.duot5 import DuoT5

        return Reranker(DuoT5(args.duot5_model))
    if name == "rv":
        from rank_llm.rerank.listwise import VicunaReranker

        return VicunaReranker()
    if name == "rz":
        from rank_llm.rerank.listwise import ZephyrReranker

        return ZephyrReranker()
    if name == "lit5":
        from rank_llm.rerank.listwise.lit5_reranker import LiT5DistillReranker

        return Reranker(LiT5DistillReranker(args.lit5_model))
    if name == "mistral":
        from rank_llm.rerank.listwise import RankListwiseOSLLM

        return Reranker(
            RankListwiseOSLLM(
                model=args.mistral_model,
                use_logits=True,
                use_alpha=True,
            )
        )
    if name == "rank_gpt":
        from rank_llm.rerank.listwise import SafeOpenai

        return Reranker(
            SafeOpenai(
                args.openai_model,
                args.context_size,
                prompt_template_path=(TEMPLATES / "rank_gpt_template.yaml"),
                keys=get_openai_api_key(),
            )
        )
    if name == "lrl":
        from rank_llm.rerank.listwise import SafeOpenai

        return Reranker(
            SafeOpenai(
                args.openai_model,
                args.context_size,
                prompt_template_path=(TEMPLATES / "rank_lrl_template.yaml"),
                keys=get_openai_api_key(),
            )
        )
    if name == "rank_gpt_apeer":
        from rank_llm.rerank.listwise import SafeOpenai

        return Reranker(
            SafeOpenai(
                args.openai_model,
                args.context_size,
                prompt_template_path=(TEMPLATES / "rank_gpt_apeer_template.yaml"),
                keys=get_openai_api_key(),
            )
        )
    if name == "gemini":
        from rank_llm.rerank.listwise import SafeGenai

        return Reranker(
            SafeGenai(
                args.gemini_model,
                args.context_size,
                keys=get_genai_api_key(),
                prompt_template_path=(TEMPLATES / "rank_gpt_apeer_template.yaml"),
                window_size=args.window_size,
                stride=args.stride,
                batch_size=args.batch_size,
                max_passage_words=args.max_passage_words,
            )
        )
    if name == "qwen":
        from rank_llm.rerank.listwise import RankListwiseOSLLM

        return Reranker(
            RankListwiseOSLLM(
                model=args.qwen_model,
                context_size=args.context_size,
                window_size=args.window_size,
                stride=args.stride,
                batch_size=args.batch_size,
                max_passage_words=args.max_passage_words,
            )
        )
    if name == "llama":
        from rank_llm.rerank.listwise import RankListwiseOSLLM

        return Reranker(
            RankListwiseOSLLM(
                model=args.llama_model,
                context_size=args.context_size,
                window_size=args.window_size,
                stride=args.stride,
                batch_size=args.batch_size,
                max_passage_words=args.max_passage_words,
            )
        )
    raise ValueError(f"Unknown reranker: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experimental reranking demos across models and datasets."
    )
    parser.add_argument("--rerankers", default=None)
    parser.add_argument("--analysis-rerankers", default=None)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--requests-file",
        default=None,
        help=(
            "Optional JSON/JSONL requests file for a single dataset. "
            "When set, skips retrieval."
        ),
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--output-dir", default="demo_outputs")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-passage-words", type=int, default=300)
    parser.add_argument("--gemini-model", default="gemini-3-flash-preview")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--monot5-model", default="castorini/monot5-3b-msmarco-10k")
    parser.add_argument("--duot5-model", default="castorini/duot5-3b-msmarco-10k")
    parser.add_argument("--lit5-model", default="castorini/LiT5-Distill-large")
    parser.add_argument("--mistral-model", default="castorini/first_mistral")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llama-model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    from rank_llm.data import DataWriter, read_requests_from_file
    from rank_llm.evaluation.trec_eval import EvalFunction
    from rank_llm.retrieve.retriever import Retriever
    from rank_llm.retrieve.topics_dict import TOPICS

    rerankers = _split_csv(args.rerankers or ",".join(DEFAULT_RERANKERS))
    datasets = _split_csv(args.datasets)
    if args.requests_file and len(datasets) != 1:
        raise ValueError("--requests-file requires exactly one --datasets value.")
    results = {}

    if not args.skip_rerank:
        for key in rerankers:
            reranker = create_reranker(key, args)
            for dataset in datasets:
                if args.requests_file:
                    retrieved_results = read_requests_from_file(args.requests_file)
                else:
                    retrieved_results = Retriever.from_dataset_with_prebuilt_index(
                        dataset, k=args.k
                    )
                if args.num_queries is not None:
                    retrieved_results = retrieved_results[: args.num_queries]
                ret_ndcg_10 = None
                if not args.skip_eval:
                    topics = TOPICS[dataset]
                    ret_ndcg_10 = EvalFunction.from_results(retrieved_results, topics)
                kwargs = {"populate_invocations_history": not args.no_history}
                rerank_results = reranker.rerank_batch(retrieved_results, **kwargs)

                writer = DataWriter(rerank_results)
                output_path_prefix = Path(args.output_dir) / dataset / key
                output_path_prefix.mkdir(parents=True, exist_ok=True)
                writer.write_in_jsonl_format(
                    str(output_path_prefix / "rerank_results.jsonl")
                )
                writer.write_in_trec_eval_format(
                    str(output_path_prefix / "rerank_results.txt")
                )
                if not args.no_history:
                    writer.write_inference_invocations_history(
                        str(output_path_prefix / "inference_invocations_history.json")
                    )

                rerank_ndcg_10 = None
                if not args.skip_eval:
                    rerank_ndcg_10 = EvalFunction.from_results(rerank_results, topics)
                    results[(key, dataset)] = (ret_ndcg_10, rerank_ndcg_10)
                    with open(output_path_prefix / "eval_results.txt", "w") as f:
                        f.write(f"{(ret_ndcg_10, rerank_ndcg_10)}")

            del reranker

        print(results)

    if args.skip_analysis or args.no_history:
        return

    from rank_llm.analysis.response_analysis import ResponseAnalyzer

    if args.analysis_rerankers:
        analysis_rerankers = _split_csv(args.analysis_rerankers)
    elif args.rerankers:
        analysis_rerankers = rerankers
    else:
        analysis_rerankers = DEFAULT_ANALYSIS_RERANKERS
    results = {}
    for model in analysis_rerankers:
        use_alpha = model == "mistral"
        paths = [
            str(
                Path(args.output_dir)
                / dataset
                / model
                / "inference_invocations_history.json"
            )
            for dataset in datasets
        ]
        try:
            analyzer = ResponseAnalyzer.from_stored_files(paths, use_alpha=use_alpha)
            error_counts = analyzer.count_errors(verbose=True, normalize=True)
            results[model] = error_counts.__repr__()
        except Exception as e:
            results[model] = f"analysis_failed: {e}"

    print(results)


if __name__ == "__main__":
    main()

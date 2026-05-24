import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.pointwise.pointwise_vllm import PointwiseVLLM

MODEL_ID = "ljw13/retro-star-qwen2.5-7b-instruct-0923"
INPUT_BASE = Path("rerank_results/converted_bright_for_rankllm")
QRELS_BASE = Path("../pyserini/tools/topics-and-qrels")

TASKS = [
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
]

CONDITIONS = [
    (
        "reason_embed_solo",
        "first_stage/reason_embed",
        "retrieve_results_bright-{task}-reason_embed_top100.jsonl",
    )
]

REASONER_TEMPLATE_PATH = (
    "src/rank_llm/rerank/prompt_templates/pointwise_reasoner_embed_template.yaml"
)
REASONER_TEMPLATE_VALUES = {
    "relevance_definition": (
        "Given a query and a document in a reasoning-intensive retrieval task, "
        "the document is relevant if it helps answer or verify the query intent."
    ),
    "query_type": "reasoning question",
    "doc_type": "retrieved passage",
}


async def _run_all(reranker: Reranker, output_root: Path) -> None:
    for condition_name, subdir, filename_template in CONDITIONS:
        for task in TASKS:
            out_path = output_root / condition_name
            out_jsonl = out_path / f"{task}_top100.jsonl"
            out_trec = out_path / f"{task}_top100.txt"
            out_inv = out_path / f"{task}_top100_inv.json"
            out_metrics = out_path / f"{task}_top100_metrics.json"
            if all(p.exists() for p in [out_jsonl, out_trec, out_inv, out_metrics]):
                print(f"[skip] {condition_name}/{task}: outputs already exist.")
                continue

            file_name = INPUT_BASE / subdir / filename_template.format(task=task)
            if not file_name.exists():
                # Fallback for flattened converted inputs.
                file_name = INPUT_BASE / filename_template.format(task=task)
            qrels = QRELS_BASE / f"qrels.bright-{task}.fixed.txt"
            retrieve_results = read_requests_from_file(str(file_name))
            retrieve_ndcg_10 = EvalFunction.from_results(retrieve_results, str(qrels))

            rerank_results = await reranker.rerank_batch_async(
                requests=retrieve_results,
                populate_invocations_history=True,
            )
            rerank_ndcg_10 = EvalFunction.from_results(rerank_results, str(qrels))

            writer = DataWriter(rerank_results)
            out_path.mkdir(parents=True, exist_ok=True)
            writer.write_in_jsonl_format(str(out_jsonl))
            writer.write_in_trec_eval_format(str(out_trec))
            writer.write_inference_invocations_history(str(out_inv))
            with open(str(out_metrics), "w") as f:
                json.dump({"retrieve": retrieve_ndcg_10, "rerank": rerank_ndcg_10}, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument(
        "--output-root",
        default="rerank_results/reasoner_embed_original",
        help="Outputs under rerank_results/<model_alias>/<condition_name>/",
    )
    args = parser.parse_args()

    coordinator = PointwiseVLLM(
        model=MODEL_ID,
        base_url=args.base_url,
        prompt_template_path=REASONER_TEMPLATE_PATH,
        batch_size=args.batch_size,
        max_concurrent_llm_calls=args.max_concurrent,
        scoring_mode="tagged_score_0_100",
        max_generation_tokens=512,
        template_values=REASONER_TEMPLATE_VALUES,
    )
    reranker = Reranker(coordinator)

    asyncio.run(_run_all(reranker=reranker, output_root=Path(args.output_root)))


if __name__ == "__main__":
    main()

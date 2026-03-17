import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

# vLLM CUDA workers must use spawn, not fork.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass  # already set, fine

# Point HF to the shared local cache so models load without re-downloading.
# vLLM paths in this codebase use HF_HOME as download_dir.
os.environ.setdefault("HF_HOME", "/home/tardis/shared/raghavv/hub")

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.pointwise.diver_reranker import DiverPointwiseReranker

MODEL_PATH = "AQ-MedAI/Diver-GroupRank-7B"
REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_BASE = REPO_ROOT / "../reranker_requests_sigir_submission"
QRELS_BASE = REPO_ROOT / "../pyan/pyserini/tools/topics-and-qrels"

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

# (condition_name, subdir_under INPUT_BASE, filename_template)
CONDITIONS = [
    (
        "diver_solo",
        "first_stage/diver",
        "retrieve_results_bright-{task}-diver_top100.jsonl",
    ),
    (
        "naf-bm25-splade",
        "naf/bm25qs_splade",
        "retrieve_results_bright-{task}-naf-bm25qs-splade_top100.jsonl",
    ),
    (
        "naf-bm25-bge",
        "naf/bm25qs_bge",
        "retrieve_results_bright-{task}-naf-bm25qs-bge_top100.jsonl",
    ),
]


def main():
    reranker = DiverPointwiseReranker(MODEL_PATH)

    for condition_name, subdir, filename_template in CONDITIONS:
        for task in TASKS:
            file_name = str(INPUT_BASE / subdir / filename_template.format(task=task))
            retrieve_results = read_requests_from_file(file_name)
            qrels = str(QRELS_BASE / f"qrels.bright-{task}.fixed.txt")
            retrieve_ndcg_10 = EvalFunction.from_results(retrieve_results, qrels)

            rerank_results = reranker.rerank_batch(requests=retrieve_results)
            rerank_ndcg_10 = EvalFunction.from_results(rerank_results, qrels)

            writer = DataWriter(rerank_results)
            out_path = Path(f"rerank_results/diver/{condition_name}")
            out_path.mkdir(parents=True, exist_ok=True)
            writer.write_in_jsonl_format(str(out_path / f"{task}_top100.jsonl"))
            writer.write_in_trec_eval_format(str(out_path / f"{task}_top100.txt"))
            with open(str(out_path / f"{task}_top100_metrics.json"), "w") as f:
                json.dump({"retrieve": retrieve_ndcg_10, "rerank": rerank_ndcg_10}, f)
            print(
                f"[{condition_name}] {task}: retrieve={retrieve_ndcg_10:.4f}, rerank={rerank_ndcg_10:.4f}"
            )


if __name__ == "__main__":
    main()

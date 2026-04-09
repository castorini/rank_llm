import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

# vLLM CUDA workers must use spawn, not fork.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass  # already set, fine

# Point HF to the shared local cache so models load without re-downloading.
# vLLM paths in this codebase use HF_HOME as download_dir.
os.environ.setdefault("HF_HOME", "/home/r2vasudeva/.cache/huggingface/hub")

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.pointwise.reason_embed_reranker import ReasonEmbedReranker

MODEL_PATH = "ljw13/retro-star-qwen3-8b-0928"
INPUT_BASE = Path("/home/r2vasudeva/reranker_requests_sigir_submission")
QRELS_BASE = Path("/home/r2vasudeva/pyserini/tools/topics-and-qrels")

# Smoke-test settings: keep this small so we can verify the pipeline cheaply.
TASKS = ["pony"]
CONDITIONS = [
    (
        "reason_embed_solo",
        "first_stage/reason_embed",
        "retrieve_results_bright-{task}-reason_embed_top100.jsonl",
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
MAX_REQUESTS = 5
RERANK_TOP_K = 20
OUTPUT_DIR = Path("rerank_results/reason_embed_smoke")


def main():
    reranker = ReasonEmbedReranker(
        MODEL_PATH,
        batch_size=8,
        relevance_definition=(
            "Given a query and a document in a reasoning-intensive retrieval task, "
            "the document is relevant if it helps answer or verify the query intent."
        ),
        query_type="reasoning question",
        doc_type="retrieved passage",
    )
    try:
        for condition_name, subdir, filename_template in CONDITIONS:
            for task in TASKS:
                file_name = str(INPUT_BASE / subdir / filename_template.format(task=task))
                retrieve_results = read_requests_from_file(file_name)[:MAX_REQUESTS]
                qrels = str(QRELS_BASE / f"qrels.bright-{task}.fixed.txt")
                retrieve_ndcg_10 = EvalFunction.from_results(retrieve_results, qrels)

                rerank_results = reranker.rerank_batch(
                    requests=retrieve_results,
                    rank_end=RERANK_TOP_K,
                    populate_invocations_history=True,
                )
                rerank_ndcg_10 = EvalFunction.from_results(rerank_results, qrels)

                out_path = OUTPUT_DIR / condition_name
                out_path.mkdir(parents=True, exist_ok=True)

                writer = DataWriter(rerank_results)
                writer.write_in_jsonl_format(
                    str(out_path / f"{task}_top{RERANK_TOP_K}_first{MAX_REQUESTS}.jsonl")
                )
                writer.write_in_trec_eval_format(
                    str(out_path / f"{task}_top{RERANK_TOP_K}_first{MAX_REQUESTS}.txt")
                )
                with open(
                    out_path / f"{task}_top{RERANK_TOP_K}_first{MAX_REQUESTS}_metrics.json",
                    "w",
                ) as f:
                    json.dump(
                        {
                            "retrieve": retrieve_ndcg_10,
                            "rerank": rerank_ndcg_10,
                            "max_requests": MAX_REQUESTS,
                            "rerank_top_k": RERANK_TOP_K,
                        },
                        f,
                    )
    finally:
        reranker.close()

if __name__ == "__main__":
    main()

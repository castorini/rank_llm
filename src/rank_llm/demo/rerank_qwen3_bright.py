import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(parent)

# --- set spawn before importing libs that might touch CUDA/vLLM ---
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass  # already set, fine

# now safe to import the rest

import json

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM

"""
Note: You need to run the vllm server with the following command:
```bash
RANK_MODEL_ID="Qwen/Qwen3-8B"
RANK_PORT=38003
RANK_VLLM_LOG="vllm_server_48003.log"
CUDA_VISIBLE_DEVICES=2 vllm serve "$RANK_MODEL_ID"  \
 --port "$RANK_PORT" \
 --dtype auto \
 --gpu-memory-utilization 0.9 \
 --max-model-len 32768 \
 --enable-prompt-tokens-details \
 --enable-prefix-caching \
 --reasoning-parser qwen3 \
 > "$RANK_VLLM_LOG" 2>&1 &
```
"""


def main():
    for fusion in ["bge-splade", "bm25-bge", "bm25-splade"]:
        for task in [
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
        ]:
            file_name = f"../bright_fusion/retrieve_results/retrieve_results_bright-{task}-rrf-{fusion}_top100.jsonl"
            retrieve_results = read_requests_from_file(file_name)
            qrels = f"../bright_fusion/qrels/qrels.bright-{task}.txt"
            retrieve_ndcg_10 = EvalFunction.from_results(retrieve_results, qrels)
            reranker = Reranker(
                RankListwiseOSLLM(
                    context_size=4096 * 4,
                    model="Qwen/Qwen3-8B",
                    use_alpha=True,
                    is_thinking=True,
                    reasoning_token_budget=4096 * 4,
                    base_url="http://localhost:38003/v1",
                )
            )
            kwargs = {"populate_invocations_history": True}
            rerank_results = reranker.rerank_batch(requests=retrieve_results, **kwargs)
            rerank_ndcg_10 = EvalFunction.from_results(rerank_results, qrels)
            writer = DataWriter(rerank_results)
            path = Path(f"../bright_fusion/rerank_results/Qwen3-8B")
            path.mkdir(parents=True, exist_ok=True)
            writer.write_in_jsonl_format(
                os.path.join(path, f"{task}-rrf-{fusion}_top100.jsonl")
            )
            writer.write_in_trec_eval_format(
                os.path.join(path, f"{task}-rrf-{fusion}_top100.txt")
            )
            writer.write_inference_invocations_history(
                os.path.join(path, f"{task}-rrf-{fusion}_top100_invocations.json")
            )
            with open(
                os.path.join(path, f"{task}-rrf-{fusion}_top100_metrics.json"), "w"
            ) as f:
                json.dump({"retrieve": retrieve_ndcg_10, "rerank": rerank_ndcg_10}, f)

            del reranker  # optional: explicit cleanup if your class supports it


if __name__ == "__main__":
    main()

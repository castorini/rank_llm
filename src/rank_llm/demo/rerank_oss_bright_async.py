"""
BRIGHT reranking demo using OpenAI-compatible vLLM (gpt-oss-20b) with **async** API.

Same inputs and outputs as ``rerank_oss_bright.py``, but each topic's requests are
reranked with ``await reranker.rerank_async(...)`` per request (via ``asyncio.gather``),
not ``rerank_batch``. That overlaps HTTP calls to the vLLM server while respecting the
listwise per-instance concurrency cap.

Requires the same vLLM server as ``rerank_oss_bright.py`` (see note below). Build the
``Reranker`` **before** ``asyncio.run`` so tokenizer / client init stays off the async loop.

Note: You need to run the vllm server with the following command:
```bash
RANK_MODEL_ID="openai/gpt-oss-20b"
RANK_PORT=48003
RANK_VLLM_LOG="vllm_server_48003.log"
CUDA_VISIBLE_DEVICES=0 vllm serve "$RANK_MODEL_ID"  \
 --port "$RANK_PORT" \
 --dtype auto \
 --gpu-memory-utilization 0.9 \
 --max-model-len 32768 \
 --enable-prompt-tokens-details \
 --enable-prefix-caching \
 > "$RANK_VLLM_LOG" 2>&1 &
```
"""

import asyncio
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, Request, Result, read_requests_from_file
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM

TEMPLATES = files("rank_llm.rerank.prompt_templates")


def build_reranker() -> Reranker:
    return Reranker(
        RankListwiseOSLLM(
            context_size=4096 * 4,
            model="openai/gpt-oss-20b",
            use_alpha=True,
            is_thinking=True,
            reasoning_token_budget=4096 * 4,
            base_url="http://localhost:48003/v1",
            prompt_template_path=(TEMPLATES / "rank_zephyr_template.yaml"),
        )
    )


async def rerank_requests_async(
    reranker: Reranker,
    requests: list[Request],
    **kwargs,
) -> list[Result]:
    """One ``rerank_async`` per request; all share one event loop and reranker instance."""
    return await asyncio.gather(
        *(reranker.rerank_async(req, **kwargs) for req in requests)
    )


async def async_main(reranker: Reranker) -> None:
    """One event loop for the whole run so the shared listwise semaphore stays valid."""
    kwargs = {"populate_invocations_history": True}

    for fusion in ["bm25-splade", "bge-splade", "bm25-bge"]:
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

            rerank_results = await rerank_requests_async(
                reranker, retrieve_results, **kwargs
            )

            rerank_ndcg_10 = EvalFunction.from_results(rerank_results, qrels)
            writer = DataWriter(rerank_results)
            path = Path("../bright_fusion/rerank_results/gpt-oss-20b-async")
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


def main() -> None:
    reranker = build_reranker()
    asyncio.run(async_main(reranker))


if __name__ == "__main__":
    main()

"""
Concurrent async reranking demo (Qwen + vLLM).

Uses multiple ``await reranker.rerank_async(...)`` calls on one event loop and one
``Reranker`` instance so sliding-window LLM work overlaps across queries.

Requires the same runtime setup as ``rerank_qwen.py`` (local vLLM with the model loaded).
"""

import asyncio
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import DataWriter, Request
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM
from rank_llm.retrieve import Retriever
from rank_llm.retrieve.topics_dict import TOPICS

# Limit how many queries run concurrently for a faster smoke test; set to None for all.
NUM_ASYNC_REQUESTS: int | None = 16


async def rerank_requests_concurrently(
    reranker: Reranker,
    requests: list[Request],
    **kwargs,
):
    """Rerank each request with ``rerank_async``; all tasks share the same event loop and reranker."""
    return await asyncio.gather(
        *(reranker.rerank_async(req, **kwargs) for req in requests)
    )


async def async_main():
    dataset_name = "dl19"
    requests = Retriever.from_dataset_with_prebuilt_index(dataset_name)
    if NUM_ASYNC_REQUESTS is not None:
        requests = requests[:NUM_ASYNC_REQUESTS]

    model_coordinator = RankListwiseOSLLM(
        model="Qwen/Qwen2.5-7B-Instruct",
    )
    reranker = Reranker(model_coordinator)
    kwargs = {"populate_invocations_history": True}

    rerank_results = await rerank_requests_concurrently(reranker, requests, **kwargs)

    analyzer = ResponseAnalyzer.from_inline_results(rerank_results, use_alpha=False)
    error_counts = analyzer.count_errors()
    print(error_counts.__repr__())

    topics = TOPICS[dataset_name]
    rerank_ndcg_10 = EvalFunction.from_results(rerank_results, topics)
    print(rerank_ndcg_10)

    writer = DataWriter(rerank_results)
    Path("demo_outputs/").mkdir(parents=True, exist_ok=True)
    writer.write_in_jsonl_format(
        "demo_outputs/rerank_results_qwen2.5-7b-instruct_async.jsonl"
    )
    writer.write_in_trec_eval_format(
        "demo_outputs/rerank_results_qwen2.5-7b-instruct_async.txt"
    )
    writer.write_inference_invocations_history(
        "demo_outputs/inference_invocations_history_qwen2.5-7b-instruct_async.json"
    )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

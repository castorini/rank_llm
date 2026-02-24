import os
import sys
from pathlib import Path

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
from rank_llm.data import DataWriter
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM
from rank_llm.retrieve import Retriever
from rank_llm.retrieve.topics_dict import TOPICS


def main():
    # By default uses BM25 for retrieval
    dataset_name = "dl19"
    requests = Retriever.from_dataset_with_prebuilt_index(dataset_name)
    model_coordinator = RankListwiseOSLLM(
        model="Qwen/Qwen2.5-7B-Instruct",
    )
    reranker = Reranker(model_coordinator)
    kwargs = {"populate_invocations_history": True}
    rerank_results = reranker.rerank_batch(requests, **kwargs)

    # Analyze the response
    analyzer = ResponseAnalyzer.from_inline_results(rerank_results, use_alpha=False)
    error_counts = analyzer.count_errors()
    print(error_counts.__repr__())

    # Eval
    topics = TOPICS[dataset_name]
    rerank_ndcg_10 = EvalFunction.from_results(rerank_results, topics)
    print(rerank_ndcg_10)

    # Write rerank results
    writer = DataWriter(rerank_results)
    Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
    writer.write_in_jsonl_format(
        f"demo_outputs/rerank_results_qwen2.5-7b-instruct.jsonl"
    )
    writer.write_in_trec_eval_format(
        f"demo_outputs/rerank_results_qwen2.5-7b-instruct.txt"
    )
    writer.write_inference_invocations_history(
        f"demo_outputs/inference_invocations_history_qwen2.5-7b-instruct.json"
    )


if __name__ == "__main__":
    main()

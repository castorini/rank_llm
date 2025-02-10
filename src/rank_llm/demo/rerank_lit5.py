import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise.lit5_reranker import (
    LiT5DistillReranker,
    LiT5ScoreReranker,
)
from rank_llm.retrieve.retriever import Retriever

dataset = "dl19"
requests = Retriever.from_dataset_with_prebuilt_index(dataset, k=100)

# Rerank multiple requests with LiT5 Distill
lit5_d_model_coordinator = LiT5DistillReranker("castorini/LiT5-Distill-large")
lit5_d_reranker = Reranker(lit5_d_model_coordinator)
kwargs = {"populate_invocations_history": True, "batch_size": 32}
rerank_results = lit5_d_reranker.rerank_batch(requests, **kwargs)
print(rerank_results)

# Rerank a single request with LiT5 Score
request = requests[0]
lit5_s_model_coordinator = LiT5ScoreReranker("castorini/LiT5-Score-large")
lit5_s_reranker = Reranker(lit5_s_model_coordinator)
kwargs = {"populate_invocations_history": True}
rerank_result = lit5_s_reranker.rerank(request, **kwargs)
print(rerank_results)

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_jsonl_format(f"demo_outputs/rerank_results.jsonl")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_inference_invocations_history(
    f"demo_outputs/inference_invocations_history.json"
)

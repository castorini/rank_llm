import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter
from rank_llm.rerank import Reranker
from rank_llm.rerank.pointwise.monot5 import MonoT5
from rank_llm.retrieve.retriever import Retriever

dataset = "dl19"
requests = Retriever.from_dataset_with_prebuilt_index(dataset, k=100)
monot5_model_coordinator = MonoT5("castorini/monot5-3b-msmarco-10k")
m_reranker = Reranker(monot5_model_coordinator)
kwargs = {"populate_exec_summary": True}
rerank_results = m_reranker.rerank_batch(requests, **kwargs)
print(rerank_results)

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_jsonl_format(f"demo_outputs/rerank_results.jsonl")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_ranking_exec_summary(f"demo_outputs/ranking_execution_summary.json")

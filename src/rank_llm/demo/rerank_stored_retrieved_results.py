import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.rerank.zephyr_reranker import ZephyrReranker
from rank_llm.retrieve.retriever import Retriever

file_name = "retrieve_results/BM25/retrieve_results_dl19.json"
retrieved_results = Retriever.from_saved_results(file_name)
reranker = ZephyrReranker()
rerank_results = reranker.rerank(retrieved_results)
print(rerank_results)

from pathlib import Path

from rank_llm.result import ResultsWriter

# write rerank results
writer = ResultsWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_json_format(f"demo_outputs/rerank_results.json")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_ranking_exec_summary(f"demo_outputs/ranking_execution_summary.json")

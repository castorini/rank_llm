import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.rerank.listwise import ZephyrReranker

file_name = "retrieve_results/BM25/retrieve_results_dl23_top20.json"
requests = read_requests_from_file(file_name)

reranker = ZephyrReranker()
rerank_results = reranker.rerank_batch(requests=requests)
print(rerank_results)

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_json_format(f"demo_outputs/rerank_results.json")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_ranking_exec_summary(f"demo_outputs/ranking_execution_summary.json")

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.retrieve.retriever import Retriever
from rank_llm.rerank.zephyr_reranker import ZephyrReranker

file_name = "retrieve_results/BM25/retrieve_results_dl19.json"
retrieved_results = Retriever.from_saved_results(file_name)
reranker = ZephyrReranker()
rerank_results = reranker.rerank(retrieved_results)
print(rerank_results)

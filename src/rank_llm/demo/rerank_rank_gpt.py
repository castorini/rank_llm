import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.rerank.api_keys import get_openai_api_key
from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.reranker import Reranker
from rank_llm.retrieve.retriever import Retriever

# By default uses BM25 for retrieval
dataset_name = "dl19"
retrieved_results = Retriever.from_dataset_with_prebuilt_index(dataset_name)
agent = SafeOpenai("gpt-3.5-turbo", 4096, keys=get_openai_api_key())
reranker = Reranker(agent)
rerank_results = reranker.rerank_batch(retrieved_results)
print(rerank_results)

from pathlib import Path

from rank_llm.data import DataWriter

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_json_format(f"demo_outputs/rerank_results.json")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_ranking_exec_summary(f"demo_outputs/ranking_execution_summary.json")

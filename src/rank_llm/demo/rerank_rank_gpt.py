import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.retrieve.retriever import Retriever
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.reranker import Reranker
from rank_llm.retrieve_and_rerank import get_api_key

# By default uses BM25 for retrieval
dataset_name = "dl19"
#retrieved_results = Retriever.from_dataset_with_prebuit_index(dataset_name)
agent = SafeOpenai("gpt-3.5-turbo", 4096, PromptMode.RANK_GPT, 0, keys=get_api_key())
reranker = Reranker(agent)
#rerank_results = reranker.rerank(retrieved_results)
#print(rerank_results)

from pathlib import Path
from rank_llm.result import ResultsWriter

# write rerank results
writer = ResultsWriter(rerank_results)
Path(f"demo_outputs/").mkdir(
    parents=True, exist_ok=True
)
writer.write_in_json_format(f"demo_outputs/rerank_results.json")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_ranking_exec_summary(f"demo_outputs/ranking_execution_summary.json")

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker, get_openai_api_key
from rank_llm.rerank.listwise import SafeOpenai, VicunaReranker, ZephyrReranker
from rank_llm.rerank.pointwise.monot5 import MonoT5
from rank_llm.retrieve.retriever import Retriever
from rank_llm.retrieve.topics_dict import TOPICS

dataset = "dl19"
requests = Retriever.from_dataset_with_prebuilt_index(dataset, k=100)
monot5_agent = MonoT5("castorini/monot5-3b-msmarco-10k")
m_reranker = Reranker(monot5_agent)
kwargs = {"populate_exec_summary": True}
rerank_results = m_reranker.rerank_batch(requests, **kwargs)
print(rerank_results)

from pathlib import Path

from rank_llm.data import DataWriter

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_json_format(f"demo_outputs/rerank_results.json")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_ranking_exec_summary(f"demo_outputs/ranking_execution_summary.json")

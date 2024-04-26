import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.api_keys import get_openai_api_key
from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.reranker import Reranker
from rank_llm.rerank.vicuna_reranker import VicunaReranker
from rank_llm.rerank.zephyr_reranker import ZephyrReranker
from rank_llm.retrieve.retriever import Retriever
from rank_llm.retrieve.topics_dict import TOPICS

# create rerankers
agent = SafeOpenai("gpt-3.5-turbo", 4096, keys=get_openai_api_key())
g_reranker = Reranker(agent)
v_reranker = VicunaReranker()
z_reranker = ZephyrReranker()
rerankers = {"rg": g_reranker, "rv": v_reranker, "rz": z_reranker}

results = {}
for dataset in ["dl19", "dl20", "dl21", "dl22", "dl23"]:
    retrieved_results = Retriever.from_dataset_with_prebuilt_index(dataset, k=20)
    topics = TOPICS[dataset]
    ret_ndcg_10 = EvalFunction.from_results(retrieved_results, topics)
    for key, reranker in rerankers.items():
        rerank_results = reranker.rerank_batch(retrieved_results)
        rerank_ndcg_10 = EvalFunction.from_results(rerank_results, topics)
        analyzer = ResponseAnalyzer.from_inline_results(rerank_results)
        error_counts = analyzer.count_errors()
        results[(key, dataset)] = (ret_ndcg_10, rerank_ndcg_10, error_counts.__repr__())
print(results)

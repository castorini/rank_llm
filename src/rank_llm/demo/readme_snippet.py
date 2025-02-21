import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import DataWriter
from rank_llm.evaluation.trec_eval import EvalFunction

# from rank_llm.rerank import Reranker, get_openai_api_key
from rank_llm.rerank.listwise import ZephyrReranker  # , SafeOpenai, VicunaReranker,
from rank_llm.retrieve.retriever import Retriever
from rank_llm.retrieve.topics_dict import TOPICS

# ------ Retrieval ------

# By default BM25 is used for retrieval of top 100 candidates.
dataset_name = "dl19"
retrieved_results = Retriever.from_dataset_with_prebuilt_index(dataset_name)

# Users can specify other retrieval methods and number of retrieved candidates.
# retrieved_results = Retriever.from_dataset_with_prebuilt_index(
#     dataset_name, RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL, k=50
# )
# -----------------------

# ------- Rerank --------

# Rank Zephyr model
reranker = ZephyrReranker()

# Rank Vicuna model
# reranker = VicunaReranker()

# RankGPT
# model_coordinator = SafeOpenai("gpt-4o-mini", 4096, keys=get_openai_api_key())
# reranker = Reranker(model_coordinator)

rerank_results = reranker.rerank_batch(requests=retrieved_results)
# -----------------------

# ----- Evaluation ------

# Evaluate retrieved results.
ndcg_10_retrieved = EvalFunction.from_results(retrieved_results, TOPICS[dataset_name])
print(ndcg_10_retrieved)

# Evaluate rerank results.
ndcg_10_rerank = EvalFunction.from_results(rerank_results, TOPICS[dataset_name])
print(ndcg_10_rerank)

# By default ndcg@10 is the eval metric, other value can be specified:
# eval_args = ["-c", "-m", "map_cut.100", "-l2"]
# map_100_rerank = EvalFunction.from_results(rerank_results, topics, eval_args)
# print(map_100_rerank)

# eval_args = ["-c", "-m", "recall.20"]
# recall_20_rerank = EvalFunction.from_results(rerank_results, topics, eval_args)
# print(recall_20_rerank)

# -----------------------

# -- Analyze invocations ---
analyzer = ResponseAnalyzer.from_inline_results(rerank_results)
error_counts = analyzer.count_errors(verbose=True)
print(error_counts)
# -----------------------

# ---- Save results ----
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_jsonl_format(f"demo_outputs/rerank_results.jsonl")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_inference_invocations_history(
    f"demo_outputs/inference_invocations_history.json"
)
# -----------------------

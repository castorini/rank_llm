import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.listwise import VicunaReranker
from rank_llm.retrieve import TOPICS, RetrievalMethod, Retriever

# By default uses BM25 for retrieval
dataset_name = "dl19"
retrieved_results = Retriever.from_dataset_with_prebuilt_index(dataset_name)

# Evaluate retrieved results.
topics = TOPICS[dataset_name]
ndcg_10 = EvalFunction.from_results(retrieved_results, topics)
print(ndcg_10)
# By default ndcg@10 is the eval metric, other value can be specified.
# map_100
eval_args = ["-c", "-m", "map_cut.100", "-l2"]
map_100 = EvalFunction.from_results(retrieved_results, topics, eval_args)
print(map_100)
# recall_20
eval_args = ["-c", "-m", "recall.20"]
recall_20 = EvalFunction.from_results(retrieved_results, topics, eval_args)
print(recall_20)

# Rerank
reranker = VicunaReranker()
kwargs = {"populate_invocations_history": True}
rerank_results = reranker.rerank_batch(retrieved_results, **kwargs)

# Analyze response:
analyzer = ResponseAnalyzer.from_inline_results(rerank_results)
error_counts = analyzer.count_errors(verbose=True)
print(error_counts)

# Evaluate rerank results.
eval_result = EvalFunction.from_results(rerank_results, TOPICS[dataset_name])
print(eval_result)

# Users can specify other retrieval methods:
retrieved_results = Retriever.from_dataset_with_prebuilt_index(
    dataset_name, RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL
)
rerank_results = reranker.rerank_batch(retrieved_results, **kwargs)

# Analyze response:
analyzer = ResponseAnalyzer.from_inline_results(rerank_results)
error_counts = analyzer.count_errors(verbose=True)

from pathlib import Path

from rank_llm.data import DataWriter

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_jsonl_format(f"demo_outputs/rerank_results.jsonl")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_inference_invocations_history(
    f"demo_outputs/inference_invocations_history.json"
)

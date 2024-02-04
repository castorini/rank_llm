import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.vicuna_reranker import VicunaReranker
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod
from rank_llm.retrieve.retriever import Retriever
from rank_llm.retrieve.topics_dict import TOPICS

# By default uses BM25 for retrieval
dataset_name = "dl19"
retrieved_results = Retriever.from_dataset_with_prebuit_index(dataset_name)

# Evaluate retrieved results.
eval_result = EvalFunction.from_results(retrieved_results, TOPICS[dataset_name])
print(eval_result)

# Rerank
reranker = VicunaReranker()
rerank_results = reranker.rerank(retrieved_results)

# Evaluate rerank results.
eval_result = EvalFunction.from_results(rerank_results, TOPICS[dataset_name])
print(eval_result)


# Users can specify other retrieval methods:
retrieved_results = Retriever.from_dataset_with_prebuit_index(
    dataset_name, RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL
)
reranker = VicunaReranker()
rerank_results = reranker.rerank(retrieved_results)

from pathlib import Path

from rank_llm.result import ResultsWriter

# write rerank results
writer = ResultsWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_json_format(f"demo_outputs/rerank_results.json")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_ranking_exec_summary(f"demo_outputs/ranking_execution_summary.json")

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.retrieve.retriever import Retriever
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod

# By default uses BM25 for retrieval
dataset_name = "dl19"
retrieved_results = Retriever.from_dataset_with_prebuit_index(dataset_name)
print(retrieved_results)
# TODO: add rerank instead of printing retrieved results

# Users can specify other retrieval methods:
retrieved_results = Retriever.from_dataset_with_prebuit_index(
    dataset_name, RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL
)
print(retrieved_results)
# TODO: add rerank instead of printing retrieved results

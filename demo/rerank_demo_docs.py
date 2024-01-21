import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from rank_llm.retrieve_and_rerank import retrieve_and_rerank
from rank_llm.retriever import RetrievalMode
from rank_llm.pyserini_retriever import RetrievalMethod

query = "What is the capital of the United States?"
docs = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
    "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
    "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
]

results = retrieve_and_rerank(
    model_path="castorini/rank_vicuna_7b_v1",
    dataset=docs,
    retrieval_mode=RetrievalMode.QUERY_AND_DOCUMENTS,
    retrieval_method=RetrievalMethod.UNSPECIFIED,
    top_k_candidates=len(docs),
    print_prompts_responses=True,
    query=query,
)

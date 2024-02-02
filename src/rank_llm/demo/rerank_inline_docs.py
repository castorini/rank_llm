import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.retrieve.retriever import Retriever
from rank_llm.rerank.zephyr_reranker import ZephyrReranker

query = "What is the capital of the United States?"
docs = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
    "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
    "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
]

retrieved_results = Retriever.from_inline_documents(query, documents=docs)
reranker = ZephyrReranker()
rerank_results = reranker.rerank(retrieved_results)
print(rerank_results)

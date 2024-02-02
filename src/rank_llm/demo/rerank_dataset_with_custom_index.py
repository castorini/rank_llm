import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.retrieve.retriever import Retriever
from rank_llm.rerank.vicuna_reranker import VicunaReranker

"""
Data prep: download the pyserini prebuilt index and topics files into any directory:

wget https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.beir-v1.0.0-fiqa.flat.20221116.505594.tar.gz
tar xzvf lucene-index.beir-v1.0.0-fiqa.flat.20221116.505594.tar.gz

wget https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.beir-v1.0.0-fiqa.test.tsv.gz
"""

retrieved_results = Retriever.from_custom_index(
    index_path="path/to/lucene-index.beir-v1.0.0-fiqa.flat.20221116.505594",
    topics_path="path/to/topics.beir-v1.0.0-fiqa.test.tsv.gz",
    index_type="lucene",
)
reranker = VicunaReranker()
rerank_results = reranker.rerank(retrieved_results)
print(rerank_results)

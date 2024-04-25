import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.rerank.vicuna_reranker import VicunaReranker
from rank_llm.retrieve.retriever import Retriever

# Data prep: download the pyserini prebuilt index and topics files
os.system(
    "wget https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.beir-v1.0.0-fiqa.flat.20221116.505594.tar.gz -P indexes/"
)
os.system(
    "tar xzvf indexes/lucene-index.beir-v1.0.0-fiqa.flat.20221116.505594.tar.gz -C indexes"
)
os.system(
    "wget https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.beir-v1.0.0-fiqa.test.tsv.gz -P topics"
)

retrieved_results = Retriever.from_custom_index(
    index_path="indexes/lucene-index.beir-v1.0.0-fiqa.flat.20221116.505594",
    topics_path="topics/topics.beir-v1.0.0-fiqa.test.tsv.gz",
    index_type="lucene",
)
reranker = VicunaReranker()
rerank_results = reranker.rerank_batch(retrieved_results)
print(rerank_results)

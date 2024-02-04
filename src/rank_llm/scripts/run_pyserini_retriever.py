import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.retrieve.pyserini_retriever import (
    PyseriniRetriever,
    RetrievalMethod,
    evaluate_retrievals,
)


def main():
    for dataset in ["dl19", "dl20", "dl21", "dl22", "news", "covid"]:
        for retrieval_method in RetrievalMethod:
            if retrieval_method == RetrievalMethod.UNSPECIFIED:
                continue
            if dataset in ["dl21", "dl22", "news", "covid"]:
                if retrieval_method not in [
                    RetrievalMethod.BM25,
                    RetrievalMethod.BM25_RM3,
                ]:
                    continue
            retriever = PyseriniRetriever(dataset, retrieval_method)
            retriever.retrieve_and_store()
    evaluate_retrievals()


if __name__ == "__main__":
    main()

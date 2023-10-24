from enum import Enum
from typing import List, Union, Dict, Any

from rank_llm.pyserini_retriever import PyseriniRetriever


class RetrievalMode(Enum):
    DATASET = "dataset"
    QUERY_AND_DOCUMENTS = "query_and_documents"
    QUERY_AND_HITS = "query_and_hits"

    def __str__(self):
        return self.value
    

class Retriever:
    def __init__(
        self, retrieval_mode: RetrievalMode
    ) -> None:
        self._retrieval_mode = retrieval_mode

    def retrieve(self, **kwargs) -> Union[None, List[Dict[str, Any]]]:
        '''
        Retriever supports three modes:

        - DATASET: args = (dataset, retrieval_method)
        - QUERY_AND_DOCUMENTS: args = (query, documents)
        - QUERY_AND_HITS: args = (query, hits)
        '''
        if self._retrieval_mode == RetrievalMode.DATASET:
            dataset = kwargs["dataset"]
            retrieval_method = kwargs["retrieval_method"]
            print(f"Retrieving with dataset {dataset}:")
            retriever = PyseriniRetriever(dataset, retrieval_method)
            # Always retrieve top 100 so that results are reusable for all top_k_candidates values.
            retriever.retrieve_and_store(k=100)
            return None
        elif self._retrieval_mode == RetrievalMode.QUERY_AND_DOCUMENTS:
            query = kwargs["query"]
            documents = kwargs["documents"]
            document_hits = []
            for passage in documents:
                document_hits.append({
                    "content": passage
                })
            retrieved_result = [{
                "query": query,
                "hits": document_hits,
            }]
            return retrieved_result
        elif self._retrieval_mode == RetrievalMode.QUERY_AND_HITS:
            query = kwargs["query"]
            hits = kwargs["hits"]
            retrieved_result = [{
                "query": query,
                "hits": hits,
            }]
            return retrieved_result
        else:
            raise ValueError(f"Invalid retrieval mode: {self._retrieval_mode}")
        
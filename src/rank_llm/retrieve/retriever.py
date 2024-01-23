from enum import Enum
from typing import List, Union, Dict, Any

from rank_llm.retrieve.pyserini_retriever import PyseriniRetriever
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod


class RetrievalMode(Enum):
    DATASET = "dataset"
    QUERY_AND_DOCUMENTS = "query_and_documents"
    QUERY_AND_HITS = "query_and_hits"

    def __str__(self):
        return self.value


class Retriever:
    def __init__(self, retrieval_mode: RetrievalMode) -> None:
        self._retrieval_mode = retrieval_mode

    def retrieve(
        self,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        retrieval_method: RetrievalMethod = RetrievalMethod.UNSPECIFIED,
        query: str = "",
    ) -> Union[None, List[Dict[str, Any]]]:
        """
        Retriever supports three modes:

        - DATASET: args = (dataset, retrieval_method)
        - QUERY_AND_DOCUMENTS: args = (dataset, query)
        - QUERY_AND_HITS: args = (dataset, query)
        """
        if self._retrieval_mode == RetrievalMode.DATASET:
            if not dataset:
                raise "Please provide name of the dataset."
            if not isinstance(dataset, str):
                raise ValueError(
                    f"Invalid dataset format: {dataset}. Expected a string representing name of the dataset."
                )
            if not retrieval_method:
                raise "Please provide a retrieval method."
            if retrieval_method == RetrievalMethod.UNSPECIFIED:
                raise ValueError(
                    f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method."
                )
            print(f"Retrieving with dataset {dataset}")
            retriever = PyseriniRetriever(dataset, retrieval_method)
            # Always retrieve top 100 so that results are reusable for all top_k_candidates values.
            retriever.retrieve_and_store(k=100)
            return None

        elif self._retrieval_mode == RetrievalMode.QUERY_AND_DOCUMENTS:
            if not dataset:
                raise "Please provide a non-empty list of documents."
            if not query or query == "":
                raise "Please provide a query string."
            document_hits = []
            for document in dataset:
                if not isinstance(document, str):
                    raise ValueError(
                        f"Invalid dataset format: {dataset}. Expected a list of strings where each string represents a document."
                    )
                document_hits.append({"content": document})
            retrieved_result = [
                {
                    "query": query,
                    "hits": document_hits,
                }
            ]
            return retrieved_result

        elif self._retrieval_mode == RetrievalMode.QUERY_AND_HITS:
            if not dataset:
                raise "Please provide a non-empty list of hits."
            for hit in dataset:
                if not isinstance(hit, Dict):
                    raise ValueError(
                        f"Invalid dataset format: {dataset}. Expected a list of Dicts where each Dict represents a hit."
                    )
            if not query or query == "":
                raise "Please provide a query string."
            retrieved_result = [
                {
                    "query": query,
                    "hits": dataset,
                }
            ]
            return retrieved_result
        else:
            raise ValueError(f"Invalid retrieval mode: {self._retrieval_mode}")

from enum import Enum
import json
from pathlib import Path
from typing import List, Union, Dict, Any

from rank_llm.retrieve.pyserini_retriever import PyseriniRetriever
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod
from rank_llm.result import Result


class RetrievalMode(Enum):
    DATASET = "dataset"
    QUERY_AND_DOCUMENTS = "query_and_documents"
    QUERY_AND_HITS = "query_and_hits"
    SAVED_FILE = "saved_file"

    def __str__(self):
        return self.value


class Retriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        retrieval_method: RetrievalMethod = RetrievalMethod.UNSPECIFIED,
        query: str = None,
    ) -> None:
        self._retrieval_mode = retrieval_mode
        self._dataset = dataset
        self._retrieval_method = retrieval_method
        self._query = query

    @staticmethod
    def from_inline_documents(query: str, documents: List[str]):
        """
        Creates a Retriever instance for inline documents with the passed in query.

        Args:
            query (str): The search query.
            documents (List[str]): A list of documents to search from.

        Returns:
            List[Dict[str, Any]]: The retrieval results.

        Raises:
            ValueError: If query or documents are invalid or missing.
        """
        if not query or query == "":
            raise ValueError("Please provide a query string.")
        if not documents:
            raise ValueError("Please provide a non-empty list of documents.")
        retriever = Retriever(
            RetrievalMode.QUERY_AND_DOCUMENTS, dataset=documents, query=query
        )
        return retriever.retrieve()

    @staticmethod
    def from_inline_hits(query: str, hits: List[Dict[str, Any]]):
        """
        Creates a Retriever instance for inline hits with the passed in query.

        Args:
            query (str): The search query.
            hits (List[Dict[str, Any]]): Predefined hits relevant to the query.

        Returns:
            List[Dict[str, Any]]: The retrieval results.

        Raises:
            ValueError: If query or hits are invalid or missing.
        """
        if not query or query == "":
            raise ValueError("Please provide a query string.")
        if not hits:
            raise ValueError("Please provide a non-empty list of hits.")

        retriever = Retriever(RetrievalMode.QUERY_AND_HITS, dataset=hits, query=query)
        return retriever.retrieve()

    @staticmethod
    def from_dataset_with_prebuit_index(
        dataset_name: str, retrieval_method: RetrievalMethod = RetrievalMethod.BM25
    ):
        """
        Creates a Retriever instance for a dataset with a prebuilt index.

        Args:
            dataset_name (str): The name of the dataset.
            retrieval_method (RetrievalMethod): The retrieval method to be used (e.g. BM25).

        Returns:
            List[Dict[str, Any]]: The retrieval results.

        Raises:
            ValueError: If dataset name or retrieval method is invalid or missing.
        """
        if not dataset_name:
            raise ValueError("Please provide name of the dataset.")
        if not isinstance(dataset_name, str):
            raise ValueError(
                f"Invalid dataset format: {dataset_name}. Expected a string representing name of the dataset."
            )
        if not retrieval_method:
            raise "Please provide a retrieval method."
        if retrieval_method == RetrievalMethod.UNSPECIFIED:
            raise ValueError(
                f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method."
            )
        retriever = Retriever(
            RetrievalMode.DATASET,
            dataset=dataset_name,
            retrieval_method=retrieval_method,
        )
        return retriever.retrieve()

    @staticmethod
    def from_saved_results(file_name: str):
        """
        Creates a Retriever instance from saved retrieval results specified by 'file_name'.

        Args:
            file_name (str): The file name containing the saved retrieval results.

        Returns:
            List[Dict[str, Any]]: The retrieval results loaded from the file.

        Raises:
            ValueError: If the file content is not in the expected format.
        """
        with open(file_name, "r") as f:
            retrieved_results = json.load(f)
        if not isinstance(retrieved_results, list):
            raise ValueError(
                f"Invalid retrieval format: Expected a list of dictionaries, got {type(retrieved_results)}"
            )
        retriever = Retriever(RetrievalMode.SAVED_FILE, dataset=retrieved_results)
        return retriever.retrieve()

    def retrieve(self) -> List[Dict[str, Any]]:
        """
        Executes the retrieval process based on the configation provided with the Retriever instance.

        Returns:
            List[Dict[str, Any]]: A list of retrieval results.

        Raises:
            ValueError: If the retrieval mode is invalid or the result format is not as expected.
        """
        if self._retrieval_mode == RetrievalMode.DATASET:
            candidates_file = Path(
                f"retrieve_results/{self._retrieval_method.name}/retrieve_results_{self._dataset}.json"
            )
            if not candidates_file.is_file():
                print(f"Retrieving with dataset {self._dataset}")
                pyserini = PyseriniRetriever(self._dataset, self._retrieval_method)
                # Always retrieve top 100 so that results are reusable for all top_k_candidates values.
                retrieved_results = pyserini.retrieve_and_store(k=100)
            else:
                print("Reusing existing retrieved results.")
                with open(candidates_file, "r") as f:
                    loaded_results = json.load(f)
                retrieved_results = [
                    Result(r["query"], r["hits"]) for r in loaded_results
                ]

        elif self._retrieval_mode == RetrievalMode.QUERY_AND_DOCUMENTS:
            document_hits = []
            for i, document in enumerate(self._dataset):
                if not isinstance(document, str):
                    raise ValueError(
                        f"Invalid dataset format: {self._dataset}. Expected a list of strings where each string represents a document."
                    )
                document_hits.append(
                    {"content": document, "qid": 1, "docid": i + 1, "rank": i + 1}
                )
            retrieved_results = [Result(self._query, document_hits)]

        elif self._retrieval_mode == RetrievalMode.QUERY_AND_HITS:
            retrieved_results = [Result(self._query, self._dataset)]
        elif self._retrieval_mode == RetrievalMode.SAVED_FILE:
            retrieved_results = [
                Result(query=r["query"], hits=r["hits"]) for r in self._dataset
            ]
        else:
            raise ValueError(f"Invalid retrieval mode: {self._retrieval_mode}")
        for result in retrieved_results:
            self._validate_result(result)
        return retrieved_results

    def _validate_result(self, result: Result):
        if not isinstance(result, Result):
            raise ValueError(
                f"Invalid result format: Expected type `Result`, got {type(result)}"
            )
        if not result.query:
            raise ValueError(f"Invalid format: missing `query`")
        if not result.hits:
            raise ValueError(f"Invalid format: missing `hits`")
        for hit in result.hits:
            if not isinstance(hit, Dict):
                raise ValueError(
                    f"Invalid hits format: Expected a list of Dicts where each Dict represents a hit."
                )
            if "content" not in hit.keys():
                raise ValueError((f"Invalid format: Missing `content` key in hit"))

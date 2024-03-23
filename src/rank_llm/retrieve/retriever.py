import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

from rank_llm.result import Result
from rank_llm.retrieve.pyserini_retriever import PyseriniRetriever, RetrievalMethod
from rank_llm.retrieve.repo_info import QUERY_INFO
from rank_llm.retrieve.util import compute_md5, download_encoded_queries


class RetrievalMode(Enum):
    DATASET = "dataset"
    QUERY_AND_DOCUMENTS = "query_and_documents"
    QUERY_AND_HITS = "query_and_hits"
    SAVED_FILE = "saved_file"
    CUSTOM = "custom"

    def __str__(self):
        return self.value


class Retriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        retrieval_method: RetrievalMethod = RetrievalMethod.UNSPECIFIED,
        query: str = None,
        index_path: str = None,
        topics_path: str = None,
        index_type: str = None,
        encoder: str = None,
        onnx: bool = False,
    ) -> None:
        self._retrieval_mode = retrieval_mode
        self._dataset = dataset
        self._retrieval_method = retrieval_method
        self._query = query
        self._index_path = index_path
        self._topics_path = topics_path
        self._index_type = index_type
        self._encoder = encoder
        self._onnx = onnx

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
    def from_dataset_with_prebuilt_index(
        dataset_name: str,
        retrieval_method: RetrievalMethod = RetrievalMethod.BM25,
        k: int = 100,
    ):
        """
        Creates a Retriever instance for a dataset with a prebuilt index.

        Args:
            dataset_name (str): The name of the dataset.
            retrieval_method (RetrievalMethod): The retrieval method to be used. Defaults to BM25.
            k (int, optional): The top k hits to retrieve. Defaults to 100.

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
        return retriever.retrieve(k=k)

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

    @staticmethod
    def from_custom_index(
        index_path: str,
        topics_path: str,
        index_type: str,
        encoder: str = None,
        onnx: bool = False,
        k: int = 100,
    ):
        """
        Creates a Retriever instance for a dataset with a prebuilt index.

        Args:
            index_path (str): The path to the lucene or impact index.
            topics_path (str): The path to the topics file.
            index_type (str): Index type; choices: [lucene, impact].
            encoder (str, optional): The encoder used in impact indexes. Defaults to None.
            onnx (bool, optional): Flag for using onnx in impact indexes. Defaults to False.
            k (int, optional): The top k hits to retrieve. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: The retrieval results.

        Raises:
            ValueError: If index_path or topics_path are invalid paths and if index_type is not lucene or impact
        """
        if not index_path:
            raise ValueError("Please provide a path to the index")
        if not topics_path:
            raise ValueError("Please provide a path to the topics file")
        if index_type not in ["lucene", "impact"]:
            raise ValueError(f"index_type must be [lucene, impact], not {index_type}")

        # implied from name of index and topic dir
        index_name = os.path.basename(os.path.normpath(index_path))
        topics_name = os.path.basename(os.path.normpath(topics_path))
        dataset_name = f"index-{index_name}_topic-{topics_name}_type-{index_type}_encoder-{encoder}_onnx-{onnx}"
        retriever = Retriever(
            retrieval_mode=RetrievalMode.CUSTOM,
            dataset=dataset_name,
            retrieval_method=RetrievalMethod.CUSTOM_INDEX,
            index_path=index_path,
            topics_path=topics_path,
            index_type=index_type,
            encoder=encoder,
            onnx=onnx,
        )
        return retriever.retrieve(k=k)

    def retrieve(
        self, retrieve_results_dirname: str = "retrieve_results", k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Executes the retrieval process based on the configation provided with the Retriever instance.

        Returns:
            List[Dict[str, Any]]: A list of retrieval results.

        Raises:
            ValueError: If the retrieval mode is invalid or the result format is not as expected.
        """
        if self._retrieval_mode == RetrievalMode.DATASET:
            candidates_file = Path(
                f"{retrieve_results_dirname}/{self._retrieval_method.name}/retrieve_results_{self._dataset}_top{k}.json"
            )
            query_name = f"{self._retrieval_method.name}/retrieve_results_{self._dataset}_top{k}.json"
            if not candidates_file.is_file():
                try:
                    file_path = download_encoded_queries(query_name)
                    with open(file_path, "r") as f:
                        loaded_results = json.load(f)
                    retrieved_results = [
                        Result(r["query"], r["hits"]) for r in loaded_results
                    ]
                except ValueError as e:
                    print(f"Retrieving with dataset {self._dataset}")
                    pyserini = PyseriniRetriever(self._dataset, self._retrieval_method)
                    retrieved_results = pyserini.retrieve_and_store(k=k)
            else:
                print("Reusing existing retrieved results.")
                md5_local = compute_md5(candidates_file)
                if QUERY_INFO[query_name]["md5"] != md5_local:
                    print("Query Cache MD5 does not match Local")
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
                    {
                        "content": document,
                        "qid": 1,
                        "docid": i + 1,
                        "rank": i + 1,
                        "score": i + 1,
                    }
                )
            retrieved_results = [Result(self._query, document_hits)]

        elif self._retrieval_mode == RetrievalMode.QUERY_AND_HITS:
            retrieved_results = [Result(self._query, self._dataset)]
        elif self._retrieval_mode == RetrievalMode.SAVED_FILE:
            retrieved_results = [
                Result(query=r["query"], hits=r["hits"]) for r in self._dataset
            ]
        elif self._retrieval_mode == RetrievalMode.CUSTOM:
            candidates_file = Path(
                f"{retrieve_results_dirname}/{self._retrieval_method.name}/retrieve_results_{self._dataset}_top{k}.json"
            )
            if not candidates_file.is_file():
                print(f"Retrieving with dataset {self._dataset}")
                pyserini = PyseriniRetriever(
                    dataset=self._dataset,
                    retrieval_method=self._retrieval_method,
                    index_path=self._index_path,
                    topics_path=self._topics_path,
                    index_type=self._index_type,
                    encoder=self._encoder,
                    onnx=self._onnx,
                )
                retrieved_results = pyserini.retrieve_and_store(k=k)
            else:
                print("Reusing existing retrieved results.")
                with open(candidates_file, "r") as f:
                    loaded_results = json.load(f)
                retrieved_results = [
                    Result(r["query"], r["hits"]) for r in loaded_results
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

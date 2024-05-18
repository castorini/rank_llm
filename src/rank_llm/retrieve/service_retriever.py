import json
import requests
from urllib import parse
from enum import Enum
from typing import Any, Dict, List, Union

from rank_llm.data import Request, Candidate, Query
from rank_llm.retrieve.pyserini_retriever import PyseriniRetriever, RetrievalMethod
from rank_llm.retrieve.repo_info import HITS_INFO
from rank_llm.retrieve.utils import compute_md5, download_cached_hits


class RetrievalMode(Enum):
    DATASET = "dataset"
    CUSTOM = "custom"

    def __str__(self):
        return self.value


class ServiceRetriever:
    def __init__(
        self,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        host: str,
        retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
        retrieval_method: RetrievalMethod = RetrievalMethod.UNSPECIFIED,
        # index_path: str = None,
        # topics_path: str = None,
        # index_type: str = None,
        # encoder: str = None,
        # onnx: bool = False,
    ) -> None:
        self._retrieval_mode = retrieval_mode
        self._host = host
        self._retrieval_method = retrieval_method
        self._dataset = dataset
        # self._index_path = index_path
        # self._topics_path = topics_path
        # self._index_type = index_type
        # self._encoder = encoder
        # self._onnx = onnx

    @staticmethod
    def from_dataset_with_prebuilt_index(
        request: Request, 
        dataset_name: str,
        retrieval_method: RetrievalMethod = RetrievalMethod.BM25,
        k: int = 100, #Anserini API currently does not support passing in k
        host: str = "http://localhost:8081",
    ):
        """
        Creates a Retriever instance for a dataset with a prebuilt index.

        Args:
            dataset_name (str): The name of the dataset.
            retrieval_method (RetrievalMethod): The retrieval method to be used. Defaults to BM25.
            k (int, optional): The top k hits to retrieve. Defaults to 100.

        Returns:
            Request. Contains a query and list of candidates
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
        retriever = ServiceRetriever(
            retrieval_mode=RetrievalMode.DATASET,
            retrieval_method=retrieval_method,
            dataset=dataset_name,
            host=host,
        )
        return retriever.retrieve(k=k, request=request)
        

    def retrieve(
        self, 
        request: Request,
        k: int = 100, # Anserini REST API currently does not accept k as a parameter
    ) -> List[Request]:
        """
        Executes the retrieval process based on the configation provided with the Retriever instance. Takes in a Request object with a query and empty candidates object. 

        Returns:
            Request. Contains a query and list of candidates
        Raises:
            ValueError: If the retrieval mode is invalid or the result format is not as expected.
        """
        if self._retrieval_mode == RetrievalMode.DATASET:
            query_url = parse.quote(request.query.text)
            url = f"{self._host}/api/collection/{self._dataset}/search?query={query_url}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                retrieved_results = Request(
                    query = Query(text = data["query"]["text"], qid = data["query"]["qid"])
                )
                collection = []
                for candidate in data["candidates"]:
                    collection.append(Candidate(
                        docid = candidate["docid"],
                        score = candidate["score"],
                        doc = candidate["doc"],
                    ))
                retrieved_results.candidates = collection
            else: 
                raise ValueError(f"Failed to retrieve data from Anserini server. Error code: {response.status_code}")
        else:
            raise ValueError(f"Invalid retrieval mode: {self._retrieval_mode}. Only DATASET mode is currently supported.")
        return retrieved_results

import json
import requests
from urllib import parse
from enum import Enum
from typing import Any, Dict, List, Union

from rank_llm.data import Request, Candidate, Query
from rank_llm.retrieve.pyserini_retriever import PyseriniRetriever, RetrievalMethod
from rank_llm.retrieve.repo_info import HITS_INFO
from rank_llm.retrieve.utils import compute_md5, download_cached_hits
from rank_llm.retrieve.retriever import RetrievalMode, Retriever


class ServiceRetriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
        retrieval_method: RetrievalMethod = RetrievalMethod.BM25,
    ) -> None:
        """
        Creates a ServiceRetriever instance with a specified retrieval method and mode.

        Args:
            retrieval_mode (RetrievalMode): The retrieval mode to be used. Defaults to DATASET. Only DATASET mode is currently supported.
            retrieval_method (RetrievalMethod): The retrieval method to be used. Defaults to BM25.

        Raises:
            ValueError: If retrieval mode or retrieval method is invalid or missing.
        """
        self._retrieval_mode = retrieval_mode
        self._retrieval_method = retrieval_method

        if retrieval_mode != RetrievalMode.DATASET:
            raise ValueError(f"{retrieval_mode} is not supported for ServiceRetriever. Only DATASET mode is currently supported.")
        
        if retrieval_method != RetrievalMethod.BM25:
            raise ValueError(f"{retrieval_method} is not supported for ServiceRetriever. Only BM25 is currently supported.")
        
        if not retrieval_method:
            raise "Please provide a retrieval method."
        
        if retrieval_method == RetrievalMethod.UNSPECIFIED:
            raise ValueError(f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method.")
        
    def retrieve(
        self, 
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        request: Request,
        k: int = 50, 
        host: str = "http://localhost:8081",
    ) -> Request:
        """
        Executes the retrieval process based on the configation provided with the Retriever instance. Takes in a Request object with a query and empty candidates object and the top k items to retrieve. 

        Args:
            request (Request): The request containing the query and qid. 
            dataset (str): The name of the dataset.
            k (int, optional): The top k hits to retrieve. Defaults to 100.
            host (str): The Anserini API host address. Defaults to http://localhost:8081

        Returns:
            Request. Contains a query and list of candidates
        Raises:
            ValueError: If the retrieval mode is invalid or the result format is not as expected.
        """

        parsed_query = parse.quote(request.query.text)
        url = f"{host}/api/collection/{dataset}/search?query={parsed_query}&hits={str(k)}&qid={request.query.qid}"

        response = requests.get(url)
        if response.ok:
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
        return retrieved_results

from urllib import parse

import requests

from rank_llm.data import Candidate, Query, Request

from . import RetrievalMethod, RetrievalMode


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
            raise ValueError(
                f"{retrieval_mode} is not supported for ServiceRetriever. Only DATASET mode is currently supported."
            )

        if retrieval_method != RetrievalMethod.BM25:
            raise ValueError(
                f"{retrieval_method} is not supported for ServiceRetriever. Only BM25 is currently supported."
            )

    def retrieve(
        self,
        dataset: str,
        request: Request,
        k: int = 50,
        host: str = "http://localhost:8081",
        timeout: int = 15
        * 60,  # downloding and decompressing the index can take a long time.
    ) -> Request:
        """
        Executes the retrieval process based on the configation provided with the Retriever instance. Takes in a Request object with a query and empty candidates object and the top k items to retrieve.

        Args:
            request (Request): The request containing the query and qid.
            dataset (str): The name of the dataset.
            k (int, optional): The top k hits to retrieve. Defaults to 100.
            host (str): The Pyserini API host address. Defaults to http://localhost:8081

        Returns:
            Request. Contains a query and list of candidates
        Raises:
            ValueError: If the retrieval mode is invalid or the result format is not as expected.
        """

        # Pyserini v2.1.0 REST API: GET /v1/{index}/search?query=&hits=
        # The index is a path-style segment, and the endpoint accepts only
        # "query" and "hits" (the qid is not part of the request or response).
        url = f"{host}/v1/{dataset}/search?query={parse.quote(request.query.text)}&hits={str(k)}"
        print(url)
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise type(e)(
                f"Failed to retrieve data from Pyserini server: {str(e)}"
            ) from e

        data = response.json()
        # The response echoes back only the query text, so carry the qid through
        # from the original request to preserve it for downstream evaluation.
        retrieved_results = Request(
            query=Query(text=data["query"]["text"], qid=request.query.qid)
        )

        for candidate in data["candidates"]:
            # The Pyserini REST API returns "doc" as a plain content string (or
            # null), while downstream prompt construction expects a dict
            # (doc["contents"]). Normalize so both shapes keep working.
            doc = candidate["doc"]
            if isinstance(doc, dict):
                normalized_doc = doc
            else:
                normalized_doc = {"contents": doc if doc is not None else ""}
            retrieved_results.candidates.append(
                Candidate(
                    docid=candidate["docid"],
                    score=candidate["score"],
                    doc=normalized_doc,
                )
            )

        return retrieved_results

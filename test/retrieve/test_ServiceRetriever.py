import unittest

from rank_llm import retrieve_and_rerank
from rank_llm.data import Candidate, Query, Request
from rank_llm.retrieve import RetrievalMethod, RetrievalMode, ServiceRetriever


class TestServiceRetriever(unittest.TestCase):
    def test_from_datatest_with_prebuilt_index(self):
        service_retriever = ServiceRetriever(
            retrieval_method=RetrievalMethod.BM25, retrieval_mode=RetrievalMode.DATASET
        )
        response = [
            service_retriever.retrieve(
                dataset="msmarco-v2.1-doc",
                request=Request(query=Query(text="hello", qid="1234")),
                k=20,
                host="http://localhost:8081",
            )
        ]

        assert len(response[0].candidates) == 20
        assert type(response[0].candidates[0]) == Candidate
        assert response[0].query == Query(text="hello", qid="1234")

    def test_retrieve_and_rerank_interactive(self):
        top_k = 14

        response = retrieve_and_rerank.retrieve_and_rerank(
            dataset="msmarco-v2.1-doc",
            query="hello",
            model_path="rank_zephyr",
            interactive=True,
            top_k_retrieve=top_k,
            exec_summary=False,
        )

        response = response[0]
        assert len(response.candidates) == top_k
        for candidate in response.candidates:
            print(candidate.docid)
            print(candidate.score)
            # print(candidate.doc)

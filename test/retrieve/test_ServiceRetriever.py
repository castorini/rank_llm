import unittest

from rank_llm.retrieve.service_retriever import ServiceRetriever
from rank_llm.data import Request, Query, Candidate
from rank_llm import retrieve_and_rerank
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod
from rank_llm.retrieve.retriever import RetrievalMode, Retriever

class TestServiceRetriever(unittest.TestCase):
    def test_from_datatest_with_prebuilt_index(self):

        response = ServiceRetriever.from_dataset_with_prebuilt_index(
            request=Request(
                query=Query(
                    text="hello",
                    qid="1234"
                )
            ), 
            dataset_name="msmarco-v2.1-doc",
            k=20,
        ) 

        assert(len(response.candidates)==20)
        assert(type(response.candidates[0]) == Candidate)
        assert(response.query == Query(text='hello', qid='1234'))

    def test_retrieve_and_rerank_interactive(self):
        top_k_candidates=1

        response = retrieve_and_rerank.retrieve_and_rerank(
            dataset="msmarco-v2.1-doc", 
            query="hello", 
            model_path="rank_zephyr",
            interactive=True,
            top_k_candidates=1,
        )

        assert(len(response.candidates)==top_k_candidates)
        for candidate in response.candidates:
            print(candidate.docid)
            print(candidate.score)
            # print(candidate.doc)


    

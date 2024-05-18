import unittest

from rank_llm.retrieve.service_retriever import ServiceRetriever
from rank_llm.data import Request, Query, Candidate


class TestServiceRetriever(unittest.TestCase):
    def test_from_datatest_with_prebuilt_index(self):

        response = ServiceRetriever.from_dataset_with_prebuilt_index(request=Request(query=Query(text="hello",qid="1234")), dataset_name="msmarco-v2.1-doc",k=20)

        assert(len(response.candidates)==20)
        assert(type(response.candidates[0]) == Candidate)
        assert(response.query == Query(text='hello', qid='1234'))

    # TODO def test_retrieve_and_rerank_interactive(self):

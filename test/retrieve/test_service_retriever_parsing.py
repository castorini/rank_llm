import unittest
from unittest.mock import MagicMock, patch

from rank_llm.data import Query, Request
from rank_llm.retrieve import RetrievalMethod, RetrievalMode, ServiceRetriever


class TestServiceRetrieverPyseriniParsing(unittest.TestCase):
    """Server-free tests for the Pyserini REST API (issue #281).

    Pyserini's search endpoint lives at /v1/{index}/search, takes no qid, and
    returns a single-field document as a plain string (or null) rather than as
    the dict the old Anserini API always returned. These tests mock the HTTP
    call so they need neither a running server nor a JDK.
    """

    def _retrieve(self, mock_get, first_doc="first passage"):
        """Run retrieve() against a mocked Pyserini search response."""
        payload = {
            "api": "v1",
            "index": "msmarco-v2.1-doc",
            "query": {"text": "hello"},
            "candidates": [
                {"docid": "d1", "score": 1.5, "rank": 1, "doc": first_doc},
                {"docid": "d2", "score": 0.5, "rank": 2, "doc": "second passage"},
            ],
        }
        mock_get.return_value = MagicMock(**{"json.return_value": payload})
        retriever = ServiceRetriever(
            retrieval_method=RetrievalMethod.BM25, retrieval_mode=RetrievalMode.DATASET
        )
        return retriever.retrieve(
            dataset="msmarco-v2.1-doc",
            request=Request(query=Query(text="hello", qid="1234")),
            k=2,
            host="http://localhost:8081",
        )

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_uses_pyserini_endpoint_and_keeps_qid(self, mock_get):
        result = self._retrieve(mock_get)

        called_url = mock_get.call_args.args[0]
        self.assertIn("/v1/msmarco-v2.1-doc/search", called_url)
        self.assertNotIn("/api/v1.0/", called_url)
        self.assertNotIn("qid=", called_url)
        # The qid is in neither the request nor the response, so the retriever
        # has to carry it over from the original request.
        self.assertEqual(result.query, Query(text="hello", qid="1234"))

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_string_doc_normalized_to_contents(self, mock_get):
        # Downstream prompt construction expects doc["contents"].
        result = self._retrieve(mock_get)

        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.candidates[0].doc, {"contents": "first passage"})
        self.assertEqual(result.candidates[1].doc, {"contents": "second passage"})

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_dict_doc_passthrough(self, mock_get):
        # A multi-field document still arrives as a dict and is left alone.
        result = self._retrieve(mock_get, first_doc={"contents": "kept", "title": "t"})

        self.assertEqual(result.candidates[0].doc, {"contents": "kept", "title": "t"})

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_null_doc_normalized_to_empty_contents(self, mock_get):
        # doc is null when the index stores no content for the document.
        result = self._retrieve(mock_get, first_doc=None)

        self.assertEqual(result.candidates[0].doc, {"contents": ""})


if __name__ == "__main__":
    unittest.main()

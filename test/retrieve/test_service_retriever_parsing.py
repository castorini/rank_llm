import unittest
from unittest.mock import MagicMock, patch

from rank_llm.data import Query, Request
from rank_llm.retrieve import RetrievalMethod, RetrievalMode, ServiceRetriever


class TestServiceRetrieverPyseriniParsing(unittest.TestCase):
    """Server-free tests for the Pyserini REST API migration (issue #281).

    The Pyserini search endpoint lives at /v1/indexes/{index}/search and returns
    each candidate's "doc" as a plain content string, unlike the old Anserini API
    which returned a dict. These tests mock the HTTP call so they need neither a
    running server nor a JDK.
    """

    def _pyserini_response(self):
        # Mirrors pyserini.server.models.Hits: query={qid,text}, doc is a str.
        return {
            "query": {"qid": "1234", "text": "hello"},
            "candidates": [
                {"docid": "d1", "score": 1.5, "doc": "first passage"},
                {"docid": "d2", "score": 0.5, "doc": "second passage"},
            ],
        }

    def _retriever(self):
        return ServiceRetriever(
            retrieval_method=RetrievalMethod.BM25, retrieval_mode=RetrievalMode.DATASET
        )

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_uses_pyserini_endpoint_and_normalizes_doc(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._pyserini_response()
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self._retriever().retrieve(
            dataset="msmarco-v2.1-doc",
            request=Request(query=Query(text="hello", qid="1234")),
            k=2,
            host="http://localhost:8081",
        )

        # Hits the new Pyserini /v1 endpoint, not the old Anserini /api/v1.0 path.
        called_url = mock_get.call_args.args[0]
        self.assertIn("/v1/indexes/msmarco-v2.1-doc/search", called_url)
        self.assertNotIn("/api/v1.0/", called_url)

        # The plain-string doc is normalized into a dict so downstream prompt
        # construction (which expects doc["contents"]) keeps working.
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.candidates[0].doc, {"contents": "first passage"})
        self.assertEqual(result.candidates[1].doc, {"contents": "second passage"})
        self.assertEqual(result.query, Query(text="hello", qid="1234"))

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_dict_doc_passthrough(self, mock_get):
        # A dict doc (legacy shape) must be passed through unchanged.
        payload = self._pyserini_response()
        payload["candidates"][0]["doc"] = {"contents": "kept", "title": "t"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self._retriever().retrieve(
            dataset="msmarco-v2.1-doc",
            request=Request(query=Query(text="hello", qid="1234")),
            k=2,
            host="http://localhost:8081",
        )

        self.assertEqual(result.candidates[0].doc, {"contents": "kept", "title": "t"})


if __name__ == "__main__":
    unittest.main()

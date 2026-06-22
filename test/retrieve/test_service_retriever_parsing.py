import unittest
from unittest.mock import MagicMock, patch

from rank_llm.data import Query, Request
from rank_llm.retrieve import RetrievalMethod, RetrievalMode, ServiceRetriever


class TestServiceRetrieverPyseriniParsing(unittest.TestCase):
    """Server-free tests for the Pyserini REST API migration (issue #281).

    The Pyserini v2.1.0 search endpoint lives at /v1/{index}/search and returns
    each candidate's "doc" as a plain content string (or null), unlike the old
    Anserini API which returned a dict. The response echoes back only the query
    text (no qid), so the retriever carries the original request's qid through.
    These tests mock the HTTP call so they need neither a running server nor a JDK.
    """

    def _pyserini_response(self):
        # Mirrors the pyserini v2.1.0 REST payload: query carries only "text",
        # each candidate exposes docid/score/rank, and doc is a plain string.
        return {
            "api": "v1",
            "index": "msmarco-v2.1-doc",
            "query": {"text": "hello"},
            "candidates": [
                {"docid": "d1", "score": 1.5, "rank": 1, "doc": "first passage"},
                {"docid": "d2", "score": 0.5, "rank": 2, "doc": "second passage"},
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

        # Hits the Pyserini v2.1.0 /v1/{index}/search endpoint, not the old
        # Anserini /api/v1.0 path nor the interim /v1/indexes/ form. The qid is
        # not part of the v2.1.0 request.
        called_url = mock_get.call_args.args[0]
        self.assertIn("/v1/msmarco-v2.1-doc/search", called_url)
        self.assertNotIn("/v1/indexes/", called_url)
        self.assertNotIn("/api/v1.0/", called_url)
        self.assertNotIn("qid=", called_url)

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

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_null_doc_normalized_to_empty_contents(self, mock_get):
        # The v2.1.0 API may return doc=null when the index stores no content;
        # normalize it to an empty-contents dict so downstream prompts don't break.
        payload = self._pyserini_response()
        payload["candidates"][0]["doc"] = None
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

        self.assertEqual(result.candidates[0].doc, {"contents": ""})


if __name__ == "__main__":
    unittest.main()

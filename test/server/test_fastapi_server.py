"""FastAPI retrieval tests with an in-process client and mocked Pyserini HTTP."""

import unittest
from importlib.util import find_spec
from unittest.mock import Mock, patch

import requests

FASTAPI_AVAILABLE = find_spec("fastapi") is not None
if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient

    from rank_llm.api.rest.app import create_app
    from rank_llm.api.rest.runtime import ServerConfig

SERVICE_PAYLOAD = {
    "dataset": "test-index",
    "query": "cats & dogs",
    "query_id": "my-query",
    "retrieval_method": "bm25",
    "retriever_host": "http://retriever.test:8081",
    "top_k_candidates": 2,
}
SERVICE_RESPONSE = {
    "query": {"text": "cats & dogs"},
    "candidates": [
        {"docid": "d1", "score": 2.0, "doc": "one"},
        {"docid": "d2", "score": 1.0, "doc": {"contents": "two", "title": "title"}},
        {"docid": "d3", "score": 0.0, "doc": None},
    ],
}


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is required")
class TestFastAPIRetrieval(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(ServerConfig(model_path="rank_identity")))

    def test_invalid_sources_and_options_never_execute(self):
        invalid = [
            {},
            {"dataset": "dl19", "requests_file": "requests.jsonl"},
            {"dataset": "dl19"},
            {"requests_file": "missing.jsonl"},
            {**SERVICE_PAYLOAD, "query": ""},
            {**SERVICE_PAYLOAD, "retriever_host": "localhost:8081"},
            {**SERVICE_PAYLOAD, "candidates": []},
            {**SERVICE_PAYLOAD, "top_k_candidates": True},
            {**SERVICE_PAYLOAD, "validate_only": True},
        ]
        with (
            patch("rank_llm.api.rest.runtime.initialize_reranker") as initialize,
            patch("rank_llm.retrieve.service_retriever.requests.get") as get,
        ):
            for payload in invalid:
                with self.subTest(payload=payload):
                    response = self.client.post("/v1/retrieve-and-rerank", json=payload)
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(response.json()["status"], "validation_error")
            initialize.assert_not_called()
            get.assert_not_called()

    def test_invalid_http_bodies_use_error_envelope(self):
        for route in ("/v1/rerank", "/v1/retrieve-and-rerank"):
            for body in ("{", "[]", "null"):
                response = self.client.post(
                    route, content=body, headers={"Content-Type": "application/json"}
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(response.json()["schema_version"], "castorini.cli.v1")

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_upstream_errors_return_502(self, get):
        get.side_effect = requests.Timeout("timed out")
        response = self.client.post("/v1/retrieve-and-rerank", json=SERVICE_PAYLOAD)
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["status"], "provider_error")

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_malformed_upstream_response_returns_502(self, get):
        get.return_value = Mock(**{"json.return_value": {"unexpected": "response"}})
        response = self.client.post("/v1/retrieve-and-rerank", json=SERVICE_PAYLOAD)
        self.assertEqual(response.status_code, 502, response.text)

    def test_unexpected_execution_error_returns_500(self):
        with patch(
            "rank_llm.api.rest.runtime.run_retrieve_and_rerank",
            side_effect=RuntimeError("broken inference"),
        ):
            response = self.client.post("/v1/retrieve-and-rerank", json=SERVICE_PAYLOAD)
        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()

"""FastAPI retrieval tests with an in-process client and mocked Pyserini HTTP."""

import asyncio
import builtins
import copy
import unittest
from unittest.mock import Mock, patch

import requests

from test.test_rerank_parity import (
    TRANSPORTS_AVAILABLE,
    cli_call,
    mcp_call,
    mcp_results,
)

if TRANSPORTS_AVAILABLE:
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


@unittest.skipUnless(TRANSPORTS_AVAILABLE, "FastAPI and FastMCP are required")
class TestFastAPIRetrieval(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(ServerConfig(model_path="rank_identity")))

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_service_results_match_transports_without_local_pyserini(self, get):
        get.return_value = Mock(
            **{"json.side_effect": lambda: copy.deepcopy(SERVICE_RESPONSE)}
        )
        original_import = builtins.__import__

        def no_pyserini(name, *args, **kwargs):
            if name == "pyserini" or name.startswith("pyserini."):
                raise AssertionError("Service retrieval must not import Pyserini")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_pyserini):
            code, cli = cli_call(
                [
                    "rerank",
                    "--model-path",
                    "rank_identity",
                    "--dataset",
                    "test-index",
                    "--query",
                    "cats & dogs",
                    "--query-id",
                    "my-query",
                    "--retrieval-method",
                    "bm25",
                    "--retriever-host",
                    "http://retriever.test:8081",
                    "--top-k-candidates",
                    "2",
                ]
            )
            self.assertEqual(code, 0, cli)
            response = self.client.post("/v1/retrieve-and-rerank", json=SERVICE_PAYLOAD)
            self.assertEqual(response.status_code, 200, response.text)
            mcp = asyncio.run(
                mcp_call(
                    "retrieve_and_rerank",
                    {**SERVICE_PAYLOAD, "model_path": "rank_identity"},
                )
            )
        expected = cli["artifacts"][0]["value"]
        self.assertEqual(response.json()["artifacts"][0]["value"], expected)
        self.assertEqual(mcp_results(mcp), expected)
        self.assertEqual(expected[0]["query"]["qid"], "my-query")
        self.assertEqual(len(expected[0]["candidates"]), 2)
        self.assertEqual(expected[0]["candidates"][0]["doc"], {"contents": "one"})
        self.assertEqual(expected[0]["candidates"][1]["doc"]["title"], "title")
        for call in get.call_args_list:
            self.assertEqual(
                call.args[0],
                "http://retriever.test:8081/v1/test-index/search?query=cats%20%26%20dogs&hits=2",
            )

    @patch("rank_llm.retrieve.service_retriever.requests.get")
    def test_retrieval_reuses_cached_model_and_respects_overrides(self, get):
        get.return_value = Mock(
            **{"json.side_effect": lambda: copy.deepcopy(SERVICE_RESPONSE)}
        )
        with patch(
            "rank_llm.rerank.Reranker.create_model_coordinator", return_value=None
        ) as factory:
            for overrides in ({}, {"top_k_rerank": 1}, {}, {"max_passage_words": 123}):
                response = self.client.post(
                    "/v1/retrieve-and-rerank",
                    json={**SERVICE_PAYLOAD, "overrides": overrides},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    len(response.json()["artifacts"][0]["value"][0]["candidates"]),
                    overrides.get("top_k_rerank", 2),
                )
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(factory.call_args.kwargs["max_passage_words"], 123)

    def test_invalid_sources_and_options_never_execute(self):
        invalid = [
            {},
            {"dataset": "dl19", "requests_file": "requests.jsonl"},
            {"dataset": "dl19"},
            {"requests_file": "missing.jsonl"},
            {**SERVICE_PAYLOAD, "query": ""},
            {**SERVICE_PAYLOAD, "retrieval_method": "splade++_ed"},
            {**SERVICE_PAYLOAD, "retriever_host": "localhost:8081"},
            {**SERVICE_PAYLOAD, "candidates": []},
            {**SERVICE_PAYLOAD, "top_k_candidates": True},
            {
                **SERVICE_PAYLOAD,
                "overrides": {"use_litellm": True, "use_openrouter": True},
            },
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
        for error in (
            requests.Timeout("timed out"),
            requests.ConnectionError("connection error"),
            requests.HTTPError("500 upstream error"),
        ):
            with self.subTest(error=error):
                get.side_effect = error
                response = self.client.post(
                    "/v1/retrieve-and-rerank", json=SERVICE_PAYLOAD
                )
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

    def test_old_flask_route_is_removed(self):
        response = self.client.get(
            "/api/model/rank_zephyr/index/test-index/8081?query=cats"
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()

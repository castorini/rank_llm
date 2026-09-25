"""Transport parity checks; no model downloads, Java, or live services."""

import asyncio
import contextlib
import copy
import io
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, patch

from rank_llm.api.cli.main import main
from rank_llm.api.options import RetrievalOptions, option_schema
from rank_llm.data import InferenceInvocation, Result

TRANSPORTS_AVAILABLE = all(
    find_spec(name) is not None for name in ("fastapi", "fastmcp")
)
if TRANSPORTS_AVAILABLE:
    from fastapi.testclient import TestClient
    from fastmcp import Client, FastMCP

    from rank_llm.api.mcp.tools import register_rankllm_tools
    from rank_llm.api.rest.app import create_app
    from rank_llm.api.rest.runtime import ServerConfig

DIRECT = {
    "query": {"text": "cats", "qid": "q1"},
    "candidates": [
        "one",
        {"text": "two"},
        {"docid": "d3", "score": 3, "doc": {"contents": "three", "title": "title"}},
    ],
}


def cli_call(args):
    stdout = io.StringIO()
    with (
        patch("rank_llm.api.cli.main.load_config", return_value=({}, None)),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        code = main(["--output", "json", *args])
    return code, json.loads(stdout.getvalue())


async def mcp_call(name, arguments):
    server = FastMCP("parity-test")
    register_rankllm_tools(server)
    async with Client(server) as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)
        return result


def mcp_results(result):
    assert not result.is_error, result.content
    return result.structured_content["result"]


@unittest.skipUnless(TRANSPORTS_AVAILABLE, "FastAPI and FastMCP are required")
class TestRerankParity(unittest.TestCase):
    def test_mcp_signatures_match_shared_options(self):
        server = FastMCP("schemas")
        register_rankllm_tools(server)
        tools = asyncio.run(server.get_tools())
        inference = option_schema()["properties"]
        retrieval = option_schema(RetrievalOptions)["properties"]
        for name, expected, direct_fields in (
            ("rerank", inference, {"query_text", "query_id", "candidates"}),
            ("retrieve_and_rerank", {**inference, **retrieval}, set()),
        ):
            properties = tools[name].parameters["properties"]
            self.assertEqual(set(properties) - direct_fields, set(expected))
            for field, definition in expected.items():
                with self.subTest(tool=name, field=field):
                    for key in ("type", "default", "minimum", "enum"):
                        if key in definition and field != "model_path":
                            self.assertEqual(properties[field][key], definition[key])

    def test_direct_identity_results_match_cli_http_and_mcp(self):
        code, cli = cli_call(
            [
                "rerank",
                "--model-path",
                "rank_identity",
                "--input-json",
                json.dumps(DIRECT),
                "--top-k-rerank",
                "2",
            ]
        )
        self.assertEqual(code, 0)
        http = TestClient(create_app(ServerConfig(model_path="rank_identity"))).post(
            "/v1/rerank", json={**DIRECT, "overrides": {"top_k_rerank": 2}}
        )
        self.assertEqual(http.status_code, 200, http.text)
        mcp = asyncio.run(
            mcp_call(
                "rerank",
                {
                    "model_path": "rank_identity",
                    "query_text": "cats",
                    "query_id": "q1",
                    "candidates": DIRECT["candidates"],
                    "top_k_rerank": 2,
                },
            )
        )
        expected = cli["artifacts"][0]["value"]
        self.assertEqual(http.json()["artifacts"][0]["value"], expected)
        self.assertEqual(mcp_results(mcp), expected)
        self.assertEqual([c["docid"] for c in expected[0]["candidates"]], ["1", "2"])
        self.assertEqual(expected[0]["query"]["qid"], "q1")

    def assert_retrieval_parity(self, payload):
        flags = [
            arg
            for name, value in payload.items()
            for arg in ("--" + name.replace("_", "-"), str(value))
        ]
        code, cli = cli_call(["rerank", "--model-path", "rank_identity", *flags])
        self.assertEqual(code, 0, cli)
        http = TestClient(create_app(ServerConfig(model_path="rank_identity"))).post(
            "/v1/retrieve-and-rerank", json=payload
        )
        self.assertEqual(http.status_code, 200, http.text)
        mcp = asyncio.run(
            mcp_call("retrieve_and_rerank", {**payload, "model_path": "rank_identity"})
        )
        expected = cli["artifacts"][0]["value"]
        self.assertEqual(http.json()["artifacts"][0]["value"], expected)
        self.assertEqual(mcp_results(mcp), expected)
        return expected

    def test_request_files_match_cli_http_and_mcp(self):
        # JSON/JSONL parsing variants are covered in test_data.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in [DIRECT, DIRECT]))
            results = self.assert_retrieval_parity(
                {"requests_file": str(path), "max_queries": 1, "top_k_candidates": 2}
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(len(results[0]["candidates"]), 2)

    def test_service_results_match_cli_http_and_mcp(self):
        from test.server.test_fastapi_server import SERVICE_PAYLOAD, SERVICE_RESPONSE

        with patch("rank_llm.retrieve.service_retriever.requests.get") as get:
            get.return_value.json.side_effect = lambda: copy.deepcopy(SERVICE_RESPONSE)
            results = self.assert_retrieval_parity(SERVICE_PAYLOAD)
        self.assertEqual(results[0]["query"]["qid"], "my-query")
        self.assertEqual(len(results[0]["candidates"]), 2)
        self.assertEqual(results[0]["candidates"][0]["doc"], {"contents": "one"})
        for call in get.call_args_list:
            self.assertEqual(
                call.args[0],
                "http://retriever.test:8081/v1/test-index/search?query=cats%20%26%20dogs&hits=2",
            )

    def test_multiple_passes_and_file_artifacts(self):
        coordinator = Mock()

        def rotate(requests, *args, **kwargs):
            return [
                Result(
                    r.query,
                    r.candidates[1:] + r.candidates[:1],
                    [InferenceInvocation("prompt", "response", 1, 2)],
                )
                for r in requests
            ]

        coordinator.rerank_batch.side_effect = rotate
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "rank_llm.rerank.Reranker.create_model_coordinator",
                return_value=coordinator,
            ),
        ):
            source = Path(directory) / "requests.jsonl"
            source.write_text(json.dumps(DIRECT) + "\n")
            output = Path(directory) / "out.jsonl"
            trec = Path(directory) / "out.trec"
            history = Path(directory) / "history.json"
            client = TestClient(create_app(ServerConfig(model_path="model")))
            response = client.post(
                "/v1/retrieve-and-rerank",
                json={
                    "requests_file": str(source),
                    "output_jsonl_file": str(output),
                    "output_trec_file": str(trec),
                    "invocations_history_file": str(history),
                    "overrides": {
                        "num_passes": 2,
                        "top_k_rerank": 1,
                        "populate_invocations_history": True,
                    },
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(coordinator.rerank_batch.call_count, 2)
            self.assertEqual(
                json.loads(output.read_text())["candidates"][0]["docid"], "d3"
            )
            self.assertIn("q1 Q0 d3 1", trec.read_text())
            self.assertEqual(
                json.loads(history.read_text())[0]["invocations_history"][0][
                    "response"
                ],
                "response",
            )

    def test_mcp_execution_does_not_coerce_invalid_option_types(self):
        with patch("rank_llm.rerank.Reranker.create_model_coordinator") as factory:
            for name, value in (
                ("num_passes", "2"),
                ("batch_size", True),
                ("use_litellm", "false"),
            ):
                payload = {
                    "model_path": "model",
                    "query_text": "cats",
                    "candidates": ["doc"],
                    name: value,
                }
                result = asyncio.run(mcp_call("rerank", payload))
                self.assertTrue(result.is_error, result.content)
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

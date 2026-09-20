"""Transport parity checks; no model downloads, Java, or live services."""

import asyncio
import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import fields
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, patch

from rank_llm.api.cli.main import build_parser, main
from rank_llm.api.options import RerankOptions, RetrievalOptions, option_schema
from rank_llm.data import Candidate, InferenceInvocation, Query, Request, Result

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
    envelopes = []
    with (
        patch("rank_llm.api.cli.main.load_config", return_value=({}, None)),
        patch("rank_llm.api.cli.main._emit_json", side_effect=envelopes.append),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        code = main(["--output", "json", *args])
    # Compare the emitted response independently of existing pipeline prints.
    assert len(envelopes) == 1
    return code, envelopes[0]


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
    def test_inference_fields_and_defaults_match_all_transports(self):
        parser = build_parser()
        cli = parser.parse_args(["rerank", "--model-path", "rank_identity"])
        http = parser.parse_args(["serve", "http", "--model-path", "rank_identity"])
        server = FastMCP("schemas")
        register_rankllm_tools(server)
        tools = asyncio.run(server.get_tools())
        inference_fields = {f.name for f in fields(RerankOptions)}
        retrieval_fields = {f.name for f in fields(RetrievalOptions)}
        self.assertEqual(
            set(vars(cli))
            - {"command", "output", "input_json", "stdin", "dry_run", "validate_only"},
            inference_fields | retrieval_fields,
        )
        self.assertEqual(
            set(tools["rerank"].parameters["properties"])
            - {"query_text", "query_id", "candidates"},
            inference_fields,
        )
        self.assertEqual(
            set(tools["retrieve_and_rerank"].parameters["properties"]),
            inference_fields | retrieval_fields,
        )
        for f in fields(RerankOptions):
            with self.subTest(option=f.name):
                expected = "rank_identity" if f.name == "model_path" else f.default
                self.assertEqual(getattr(cli, f.name), expected)
                self.assertEqual(getattr(http, f.name), expected)
                self.assertEqual(
                    getattr(ServerConfig(model_path="rank_identity"), f.name), expected
                )
                for name in ("rerank", "retrieve_and_rerank"):
                    props = tools[name].parameters["properties"]
                    self.assertIn(f.name, props)
                    if f.name != "model_path":
                        self.assertEqual(props[f.name].get("default"), f.default)

        retrieval_props = tools["retrieve_and_rerank"].parameters["properties"]
        for f in fields(RetrievalOptions):
            self.assertIn(f.name, retrieval_props)
            self.assertEqual(retrieval_props[f.name].get("default"), f.default)

        openapi = create_app(ServerConfig()).openapi()
        for path in ("/v1/rerank", "/v1/retrieve-and-rerank"):
            body = openapi["paths"][path]["post"]["requestBody"]["content"][
                "application/json"
            ]["schema"]
            expected = option_schema()
            # FastAPI omits null-valued schema annotations.
            expected["properties"]["reasoning_effort"].pop("default")
            self.assertEqual(body["properties"]["overrides"], expected)

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

    def test_request_files_match_cli_http_and_mcp(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("jsonl", "json"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"requests.{suffix}"
                    records = [DIRECT, {**DIRECT, "query": "dogs"}]
                    path.write_text(
                        json.dumps(records)
                        if suffix == "json"
                        else "\n".join(json.dumps(r) for r in records)
                    )
                    code, cli = cli_call(
                        [
                            "rerank",
                            "--model-path",
                            "rank_identity",
                            "--requests-file",
                            str(path),
                            "--max-queries",
                            "1",
                            "--top-k-candidates",
                            "2",
                        ]
                    )
                    self.assertEqual(code, 0, cli)
                    payload = {
                        "requests_file": str(path),
                        "max_queries": 1,
                        "top_k_candidates": 2,
                    }
                    http = TestClient(
                        create_app(ServerConfig(model_path="rank_identity"))
                    ).post("/v1/retrieve-and-rerank", json=payload)
                    self.assertEqual(http.status_code, 200, http.text)
                    mcp = asyncio.run(
                        mcp_call(
                            "retrieve_and_rerank",
                            {**payload, "model_path": "rank_identity"},
                        )
                    )
                    expected = cli["artifacts"][0]["value"]
                    self.assertEqual(http.json()["artifacts"][0]["value"], expected)
                    self.assertEqual(mcp_results(mcp), expected)
                    self.assertEqual(len(expected), 1)
                    self.assertEqual(len(expected[0]["candidates"]), 2)

    def test_dataset_results_match_cli_http_and_mcp(self):
        requests = [
            Request(Query("cats", "q1"), [Candidate("d1", 1, {"contents": "doc"})])
        ]
        with patch(
            "rank_llm.retrieve.retriever.Retriever.from_dataset_with_prebuilt_index",
            side_effect=lambda **kwargs: copy.deepcopy(requests),
        ) as retrieve:
            code, cli = cli_call(
                [
                    "rerank",
                    "--model-path",
                    "rank_identity",
                    "--dataset",
                    "dl19",
                    "--retrieval-method",
                    "bm25",
                ]
            )
            self.assertEqual(code, 0, cli)
            payload = {"dataset": "dl19", "retrieval_method": "bm25"}
            http = TestClient(
                create_app(ServerConfig(model_path="rank_identity"))
            ).post("/v1/retrieve-and-rerank", json=payload)
            mcp = asyncio.run(
                mcp_call(
                    "retrieve_and_rerank", {**payload, "model_path": "rank_identity"}
                )
            )
            self.assertEqual(http.status_code, 200, http.text)
            self.assertEqual(
                http.json()["artifacts"][0]["value"], cli["artifacts"][0]["value"]
            )
            self.assertEqual(mcp_results(mcp), cli["artifacts"][0]["value"])
            self.assertEqual(retrieve.call_count, 3)

    def test_new_options_reach_model_and_inference(self):
        coordinator = Mock()
        coordinator.rerank_batch.return_value = []
        for options in (
            {"use_litellm": True},
            {"pointwise_vllm": True},
            {"listwise_vllm_with_openai_sdk": True, "base_url": "http://model.test/v1"},
        ):
            with (
                self.subTest(options=options),
                patch(
                    "rank_llm.rerank.Reranker.create_model_coordinator",
                    return_value=coordinator,
                ) as factory,
            ):
                options = {
                    **options,
                    "reasoning_effort": "high",
                    "max_passage_words": 123,
                }
                response = asyncio.run(
                    mcp_call(
                        "rerank",
                        {
                            "model_path": "model",
                            "query_text": "cats",
                            "candidates": ["doc"],
                            **options,
                        },
                    )
                )
                self.assertFalse(response.is_error, response.content)
                for name, value in options.items():
                    self.assertEqual(factory.call_args.kwargs[name], value)
                    self.assertEqual(
                        coordinator.rerank_batch.call_args.kwargs[name], value
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

    def test_servers_expose_execution_only(self):
        server = FastMCP("tools")
        register_rankllm_tools(server)
        self.assertEqual(
            set(asyncio.run(server.get_tools())), {"rerank", "retrieve_and_rerank"}
        )
        client = TestClient(create_app(ServerConfig(model_path="model")))
        schemas = client.get("/openapi.json").json()
        with patch("rank_llm.rerank.Reranker.create_model_coordinator") as factory:
            for route, payload in (
                ("/v1/rerank", DIRECT),
                (
                    "/v1/retrieve-and-rerank",
                    {"dataset": "dl19", "retrieval_method": "bm25"},
                ),
            ):
                properties = schemas["paths"][route]["post"]["requestBody"]["content"][
                    "application/json"
                ]["schema"]["properties"]
                for flag in ("dry_run", "validate_only"):
                    self.assertNotIn(flag, properties)
                    for value in (True, False):
                        response = client.post(route, json={**payload, flag: value})
                        self.assertEqual(response.status_code, 400, response.text)
            factory.assert_not_called()

    def test_invalid_inputs_fail_before_model_initialization(self):
        invalid = [
            {**DIRECT, "candidates": "doc"},
            {**DIRECT, "candidates": [{}]},
            {**DIRECT, "query": {"qid": "q1"}},
            {**DIRECT, "dataset": "dl19"},
            {**DIRECT, "overrides": {"num_passes": 0}},
            {**DIRECT, "overrides": {"listwise_vllm_with_openai_sdk": True}},
            {
                **DIRECT,
                "overrides": {
                    "pointwise_vllm": True,
                    "listwise_vllm_with_openai_sdk": True,
                    "base_url": "http://model.test",
                },
            },
        ]
        with patch("rank_llm.rerank.Reranker.create_model_coordinator") as factory:
            client = TestClient(create_app(ServerConfig(model_path="model")))
            for payload in invalid:
                result = client.post("/v1/rerank", json=payload)
                self.assertEqual(result.status_code, 400, result.text)
            factory.assert_not_called()

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

    def test_mcp_retrieval_forwards_new_options(self):
        with patch(
            "rank_llm.retrieve_and_rerank.retrieve_and_rerank", return_value=[]
        ) as runner:
            for options in (
                {"use_litellm": True},
                {"pointwise_vllm": True},
                {
                    "listwise_vllm_with_openai_sdk": True,
                    "base_url": "http://model.test/v1",
                },
            ):
                options = {
                    **options,
                    "reasoning_effort": "high",
                    "max_passage_words": 123,
                }
                result = asyncio.run(
                    mcp_call(
                        "retrieve_and_rerank",
                        {
                            "model_path": "model",
                            "dataset": "dl19",
                            "retrieval_method": "bm25",
                            **options,
                        },
                    )
                )
                self.assertFalse(result.is_error, result.content)
                for name, value in options.items():
                    self.assertEqual(runner.call_args.kwargs[name], value)

    def test_cli_rejects_multiple_sources_before_execution(self):
        with patch("rank_llm.rerank.Reranker.create_model_coordinator") as factory:
            for flags in (
                ["--dataset", "dl19"],
                ["--requests-file", "requests.jsonl"],
                ["--retriever-host", "http://retriever.test"],
            ):
                code, response = cli_call(
                    [
                        "rerank",
                        "--model-path",
                        "model",
                        "--input-json",
                        json.dumps(DIRECT),
                        *flags,
                    ]
                )
                self.assertEqual(code, 2)
                self.assertEqual(response["status"], "validation_error")
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

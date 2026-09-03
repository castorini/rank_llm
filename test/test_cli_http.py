import contextlib
import io
import json
import subprocess
import unittest
from importlib.util import find_spec
from pathlib import Path
from shutil import which
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
FASTAPI_AVAILABLE = find_spec("fastapi") is not None


class ProviderError(Exception):
    pass


ProviderError.__module__ = "openai"

if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient

    from rank_llm.api.app import create_app
    from rank_llm.api.runtime import ServerConfig


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is required for HTTP route tests")
class TestCLIHTTP(unittest.TestCase):
    def test_healthz_route(self):
        client = TestClient(create_app(ServerConfig(model_path="model")))

        response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_rerank_route_returns_envelope(self):
        with (
            patch("rank_llm.api.runtime.initialize_reranker"),
            patch(
                "rank_llm.api.runtime.run_mcp_rerank",
                return_value=[{"query": {"text": "cats"}, "candidates": []}],
            ) as mocked,
        ):
            client = TestClient(create_app(ServerConfig(model_path="model")))
            response = client.post(
                "/v1/rerank",
                json={"query": "cats", "candidates": ["doc one"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mocked.call_args.kwargs["model_path"], "model")
        payload = response.json()
        self.assertEqual(payload["command"], "rerank")
        self.assertEqual(payload["artifacts"][0]["name"], "rerank-results")

    def test_rerank_route_applies_request_overrides(self):
        reranker = Mock()

        with (
            patch(
                "rank_llm.api.runtime.initialize_reranker",
                return_value=reranker,
            ) as initialize_reranker,
            patch(
                "rank_llm.api.runtime.run_mcp_rerank",
                return_value=[{"query": {"text": "cats"}, "candidates": []}],
            ) as mocked,
        ):
            client = TestClient(create_app(ServerConfig(model_path="model")))
            response = client.post(
                "/v1/rerank",
                json={
                    "query": "cats",
                    "candidates": ["doc one"],
                    "overrides": {
                        "model_path": "other-model",
                        "top_k_rerank": 5,
                        "reasoning_effort": "medium",
                        "use_litellm": True,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        effective_config = initialize_reranker.call_args.args[1]
        self.assertEqual(effective_config.model_path, "other-model")
        self.assertEqual(effective_config.top_k_rerank, 5)
        self.assertEqual(effective_config.reasoning_effort, "medium")
        self.assertTrue(effective_config.use_litellm)
        self.assertEqual(mocked.call_args.kwargs["model_path"], "other-model")
        self.assertEqual(mocked.call_args.kwargs["top_k_rerank"], 5)
        self.assertEqual(mocked.call_args.kwargs["reasoning_effort"], "medium")
        self.assertTrue(mocked.call_args.kwargs["use_litellm"])
        self.assertIs(mocked.call_args.kwargs["reranker"], reranker)

    def test_initialize_reranker_forwards_litellm_config(self):
        from rank_llm.api.runtime import initialize_reranker

        config = ServerConfig(model_path="openai/gpt-4o-mini", use_litellm=True)

        with patch("rank_llm.api.runtime.Reranker") as reranker_class:
            reranker_class.create_model_coordinator.return_value = object()
            initialize_reranker(config)

        kwargs = reranker_class.create_model_coordinator.call_args.kwargs
        self.assertTrue(kwargs["use_litellm"])

    def test_serve_http_with_litellm_reaches_app_creation(self):
        from rank_llm.cli.main import main

        with (
            patch("rank_llm.api.app.create_app", return_value=object()) as create_app,
            patch("uvicorn.run") as uvicorn_run,
        ):
            return_code = main(
                ["serve", "http", "--model-path", "model", "--use-litellm"]
            )

        self.assertEqual(return_code, 0)
        self.assertTrue(create_app.call_args.args[0].use_litellm)
        uvicorn_run.assert_called_once()

    def test_serve_http_rejects_conflicting_backend_flags_at_startup(self):
        from rank_llm.cli.main import main

        stdout = io.StringIO()
        with (
            patch("rank_llm.api.app.create_app") as create_app,
            patch("uvicorn.run") as uvicorn_run,
            contextlib.redirect_stdout(stdout),
        ):
            return_code = main(
                [
                    "--output",
                    "json",
                    "serve",
                    "http",
                    "--model-path",
                    "model",
                    "--use-litellm",
                    "--use-openrouter",
                ]
            )

        self.assertEqual(return_code, 2)
        create_app.assert_not_called()
        uvicorn_run.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")
        self.assertIn(
            "backend selectors cannot be combined", payload["errors"][0]["message"]
        )

    def test_rerank_route_reuses_initialized_reranker(self):
        reranker = Mock()
        reranker.get_model_coordinator.return_value = object()
        reranker.rerank_batch.return_value = []

        with patch("rank_llm.api.runtime.Reranker") as reranker_class:
            reranker_class.create_model_coordinator.return_value = object()
            reranker_class.return_value = reranker
            client = TestClient(create_app(ServerConfig(model_path="model")))
            first = client.post(
                "/v1/rerank", json={"query": "cats", "candidates": ["doc one"]}
            )
            second = client.post(
                "/v1/rerank", json={"query": "dogs", "candidates": ["doc two"]}
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(reranker_class.create_model_coordinator.call_count, 1)
        self.assertEqual(reranker_class.call_count, 1)
        self.assertEqual(reranker.rerank_batch.call_count, 2)

    def test_rerank_route_caches_rerankers_by_effective_config(self):
        reranker = Mock()
        reranker.get_model_coordinator.return_value = object()
        reranker.rerank_batch.return_value = []
        alternate_reranker = Mock()
        alternate_reranker.get_model_coordinator.return_value = object()
        alternate_reranker.rerank_batch.return_value = []

        with patch("rank_llm.api.runtime.Reranker") as reranker_class:
            reranker_class.create_model_coordinator.side_effect = [object(), object()]
            reranker_class.side_effect = [reranker, alternate_reranker]
            client = TestClient(create_app(ServerConfig(model_path="model")))
            first = client.post(
                "/v1/rerank", json={"query": "cats", "candidates": ["doc one"]}
            )
            second = client.post(
                "/v1/rerank",
                json={
                    "query": "dogs",
                    "candidates": ["doc two"],
                    "overrides": {"model_path": "other-model"},
                },
            )
            third = client.post(
                "/v1/rerank", json={"query": "mice", "candidates": ["doc three"]}
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 200)
        self.assertEqual(reranker_class.create_model_coordinator.call_count, 2)
        self.assertEqual(reranker_class.call_count, 2)
        self.assertEqual(reranker.rerank_batch.call_count, 2)
        self.assertEqual(alternate_reranker.rerank_batch.call_count, 1)

    def test_rerank_route_returns_400_for_invalid_payload(self):
        with patch("rank_llm.api.runtime.initialize_reranker"):
            client = TestClient(create_app(ServerConfig(model_path="model")))
            response = client.post("/v1/rerank", json={"query": "cats"})

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["status"], "validation_error")

    def test_rerank_route_returns_400_for_invalid_overrides(self):
        client = TestClient(create_app(ServerConfig(model_path="model")))
        response = client.post(
            "/v1/rerank",
            json={
                "query": "cats",
                "candidates": ["doc one"],
                "overrides": {
                    "use_azure_openai": True,
                    "use_openrouter": True,
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["status"], "validation_error")

    def test_rerank_route_rejects_conflicting_litellm_backends(self):
        cases = [
            (
                ServerConfig(model_path="model"),
                {"use_openrouter": True, "use_litellm": True},
            ),
            (
                ServerConfig(model_path="model"),
                {"use_azure_openai": True, "use_litellm": True},
            ),
            (
                ServerConfig(model_path="model", use_openrouter=True),
                {"use_litellm": True},
            ),
            (
                ServerConfig(model_path="model", use_litellm=True),
                {"use_azure_openai": True},
            ),
            (
                ServerConfig(model_path="model", use_openrouter=True, use_litellm=True),
                {},
            ),
        ]

        for config, overrides in cases:
            with self.subTest(config=config, overrides=overrides):
                client = TestClient(create_app(config))
                response = client.post(
                    "/v1/rerank",
                    json={
                        "query": "cats",
                        "candidates": ["doc one"],
                        "overrides": overrides,
                    },
                )

                self.assertEqual(response.status_code, 400)
                payload = response.json()
                self.assertEqual(payload["status"], "validation_error")
                self.assertIn(
                    "backend selectors cannot be combined",
                    payload["errors"][0]["message"],
                )

    def test_rerank_route_returns_400_for_invalid_override_types(self):
        client = TestClient(create_app(ServerConfig(model_path="model")))
        response = client.post(
            "/v1/rerank",
            json={
                "query": "cats",
                "candidates": ["doc one"],
                "overrides": {
                    "use_openrouter": "false",
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["status"], "validation_error")
        self.assertIn(
            "override 'use_openrouter' must be a boolean",
            payload["errors"][0]["message"],
        )

    def test_rerank_route_returns_400_for_invalid_reasoning_effort(self):
        client = TestClient(create_app(ServerConfig(model_path="model")))
        response = client.post(
            "/v1/rerank",
            json={
                "query": "cats",
                "candidates": ["doc one"],
                "overrides": {
                    "reasoning_effort": "max",
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["status"], "validation_error")
        self.assertIn(
            "override 'reasoning_effort' must be one of: high, low, medium, minimal, none, xhigh",
            payload["errors"][0]["message"],
        )

    def test_rerank_route_returns_500_for_runtime_error(self):
        with (
            patch("rank_llm.api.runtime.initialize_reranker"),
            patch(
                "rank_llm.api.runtime.run_mcp_rerank",
                side_effect=RuntimeError("boom"),
            ),
        ):
            client = TestClient(create_app(ServerConfig(model_path="model")))
            response = client.post(
                "/v1/rerank",
                json={"query": "cats", "candidates": ["doc one"]},
            )

        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["status"], "runtime_error")

    def test_rerank_route_returns_502_for_provider_error(self):
        with (
            patch("rank_llm.api.runtime.initialize_reranker"),
            patch(
                "rank_llm.api.runtime.run_mcp_rerank",
                side_effect=ProviderError("Rate limit exceeded"),
            ),
        ):
            client = TestClient(create_app(ServerConfig(model_path="model")))
            response = client.post(
                "/v1/rerank",
                json={"query": "cats", "candidates": ["doc one"]},
            )

        self.assertEqual(response.status_code, 502)
        payload = response.json()
        self.assertEqual(payload["status"], "provider_error")
        self.assertTrue(payload["errors"][0]["retryable"])

    def test_console_entrypoint_serve_http_help_resolves(self):
        cli = which("rank-llm")
        self.assertIsNotNone(cli, msg="rank-llm is not installed in PATH")

        help_result = subprocess.run(
            [cli, "serve", "http", "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(help_result.returncode, 0, msg=help_result.stderr)
        self.assertIn("Start the RankLLM HTTP server", help_result.stdout)


if __name__ == "__main__":
    unittest.main()

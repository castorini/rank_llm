import contextlib
import io
import json
import unittest
from importlib.util import find_spec
from unittest.mock import Mock, patch

FASTAPI_AVAILABLE = find_spec("fastapi") is not None


if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient

    from rank_llm.api.rest.app import create_app
    from rank_llm.api.rest.runtime import ServerConfig


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is required for HTTP route tests")
class TestCLIHTTP(unittest.TestCase):
    def test_healthz_route(self):
        client = TestClient(create_app(ServerConfig(model_path="model")))

        response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_serve_http_with_litellm_reaches_app_creation(self):
        from rank_llm.api.cli.main import main

        with (
            patch(
                "rank_llm.api.rest.app.create_app", return_value=object()
            ) as create_app,
            patch("uvicorn.run") as uvicorn_run,
        ):
            return_code = main(
                ["serve", "http", "--model-path", "model", "--use-litellm"]
            )

        self.assertEqual(return_code, 0)
        self.assertTrue(create_app.call_args.args[0].use_litellm)
        uvicorn_run.assert_called_once()

    def test_serve_http_rejects_conflicting_backend_flags_at_startup(self):
        from rank_llm.api.cli.main import main

        stdout = io.StringIO()
        with (
            patch("rank_llm.api.rest.app.create_app") as create_app,
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

    def test_rerank_route_caches_models_independently_of_execution_options(self):
        coordinator = Mock()
        coordinator.rerank_batch.return_value = []
        with patch(
            "rank_llm.rerank.Reranker.create_model_coordinator",
            return_value=coordinator,
        ) as factory:
            client = TestClient(create_app(ServerConfig(model_path="model")))
            for overrides in (
                {},
                {
                    "top_k_rerank": 1,
                    "num_passes": 2,
                    "shuffle_candidates": True,
                    "populate_invocations_history": True,
                },
                {
                    "model_path": "other-model",
                    "use_litellm": True,
                    "reasoning_effort": "medium",
                    "max_passage_words": 120,
                },
                {},
            ):
                response = client.post(
                    "/v1/rerank",
                    json={
                        "query": "cats",
                        "candidates": ["doc"],
                        "overrides": overrides,
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(factory.call_args.args[0], "other-model")
            self.assertTrue(factory.call_args.kwargs["use_litellm"])
            self.assertEqual(factory.call_args.kwargs["reasoning_effort"], "medium")
            self.assertEqual(factory.call_args.kwargs["max_passage_words"], 120)
            self.assertEqual(coordinator.rerank_batch.call_count, 5)

    def test_invalid_inputs_fail_before_model_initialization(self):
        direct = {"query": "cats", "candidates": ["doc"]}
        invalid = [
            {"query": "cats"},
            {**direct, "candidates": [{}]},
            {**direct, "dataset": "dl19"},
            {**direct, "dry_run": False},
            {**direct, "overrides": {"unsupported": True}},
            {**direct, "overrides": {"use_litellm": "false"}},
            {**direct, "overrides": {"reasoning_effort": "max"}},
            {**direct, "overrides": {"num_passes": 0}},
            {**direct, "overrides": {"pointwise_vllm": True}},
            {**direct, "overrides": {"use_litellm": True, "use_openrouter": True}},
        ]
        with patch("rank_llm.rerank.Reranker.create_model_coordinator") as factory:
            client = TestClient(create_app(ServerConfig(model_path="model")))
            for payload in invalid:
                result = client.post("/v1/rerank", json=payload)
                self.assertEqual(result.status_code, 400, result.text)
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

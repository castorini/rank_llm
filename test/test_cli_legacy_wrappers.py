import argparse
import unittest
from unittest.mock import Mock, patch

from rank_llm.scripts import (
    generate_retrieve_results_json_cache,
    run_rank_llm,
    run_response_analysis,
    run_trec_eval,
)
from rank_llm.server.flask import api as flask_api
from rank_llm.server.mcp import mcp_rankllm


class TestCLILegacyWrappers(unittest.TestCase):
    def test_run_rank_llm_wrapper_translates_snake_case_flags(self):
        with patch("rank_llm.scripts.run_rank_llm.cli_main", return_value=0) as mocked:
            exit_code = run_rank_llm.main(
                [
                    "--model_path",
                    "model",
                    "--requests_file",
                    "requests.jsonl",
                    "--output_jsonl_file",
                    "out.jsonl",
                    "--use_azure_openai",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "rerank",
                "--model-path",
                "model",
                "--requests-file",
                "requests.jsonl",
                "--output-jsonl-file",
                "out.jsonl",
                "--use-azure-openai",
            ],
        )

    def test_run_trec_eval_wrapper_delegates_to_evaluate(self):
        with patch("rank_llm.scripts.run_trec_eval.cli_main", return_value=0) as mocked:
            exit_code = run_trec_eval.main(
                argparse.Namespace(
                    model_name="model",
                    context_size=4096,
                    rerank_results_dirname="rerank_results",
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "evaluate",
                "--model-name",
                "model",
                "--context-size",
                "4096",
                "--rerank-results-dirname",
                "rerank_results",
            ],
        )

    def test_run_response_analysis_wrapper_delegates_to_analyze(self):
        with patch(
            "rank_llm.scripts.run_response_analysis.cli_main",
            return_value=0,
        ) as mocked:
            exit_code = run_response_analysis.main(
                argparse.Namespace(files=["one.json"], verbose=True)
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            mocked.call_args.args[0],
            ["analyze", "--files", "one.json", "--verbose"],
        )

    def test_generate_retrieve_results_wrapper_delegates_to_retrieve_cache(self):
        with patch(
            "rank_llm.scripts.generate_retrieve_results_json_cache.cli_main",
            return_value=0,
        ) as mocked:
            exit_code = generate_retrieve_results_json_cache.main(
                [
                    "--trec_file",
                    "run.trec",
                    "--collection_file",
                    "collection.tsv",
                    "--query_file",
                    "queries.tsv",
                    "--output_file",
                    "cache.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "retrieve-cache",
                "--trec-file",
                "run.trec",
                "--collection-file",
                "collection.tsv",
                "--query-file",
                "queries.tsv",
                "--output-file",
                "cache.json",
            ],
        )

    def test_run_rank_llm_wrapper_rejects_conflicting_backend_flags(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            run_rank_llm.main(
                [
                    "--model_path",
                    "model",
                    "--requests_file",
                    "requests.jsonl",
                    "--sglang_batched",
                    "--tensorrt_batched",
                ]
            )

    def test_flask_legacy_entrypoint_runs_flask_app(self):
        app = Mock()
        with patch(
            "rank_llm.server.flask.api.create_app",
            return_value=(app, 8082),
        ) as mocked:
            exit_code = flask_api.main(["--model", "rank_zephyr", "--port", "8082"])

        self.assertEqual(exit_code, 0)
        mocked.assert_called_once_with("rank_zephyr", 8082, False)
        app.run.assert_called_once_with(
            host="0.0.0.0",
            port=8082,
            debug=False,
        )

    def test_mcp_legacy_entrypoint_delegates_to_serve_mcp(self):
        with patch(
            "rank_llm.server.mcp.mcp_rankllm.cli_main", return_value=0
        ) as mocked:
            exit_code = mcp_rankllm.main(["--transport", "http", "--port", "9000"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            mocked.call_args.args[0],
            ["serve", "mcp", "--transport", "http", "--port", "9000"],
        )


if __name__ == "__main__":
    unittest.main()

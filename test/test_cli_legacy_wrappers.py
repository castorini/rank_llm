import argparse
import unittest
import warnings
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

    def test_run_rank_llm_wrapper_warns_and_drops_deprecated_prompt_mode(self):
        with patch("rank_llm.scripts.run_rank_llm.cli_main", return_value=0) as mocked:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                exit_code = run_rank_llm.main(
                    [
                        "--model_path",
                        "model",
                        "--prompt_mode",
                        "rank_GPT",
                        "--requests_file",
                        "requests.jsonl",
                    ]
                )

        self.assertEqual(exit_code, 0)
        prompt_mode_warnings = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "prompt_mode" in str(w.message)
        ]
        self.assertEqual(len(prompt_mode_warnings), 1)
        # The deprecated flag is still accepted but never forwarded to the CLI.
        self.assertNotIn("--prompt-mode", mocked.call_args.args[0])
        self.assertNotIn("rank_GPT", mocked.call_args.args[0])
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "rerank",
                "--model-path",
                "model",
                "--requests-file",
                "requests.jsonl",
            ],
        )

    def test_run_rank_llm_wrapper_warns_on_deprecated_prompt_mode_namespace(self):
        with patch("rank_llm.scripts.run_rank_llm.cli_main", return_value=0):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_rank_llm.main(
                    argparse.Namespace(
                        model_path="model",
                        prompt_mode="rank_GPT",
                    )
                )

        prompt_mode_warnings = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "prompt_mode" in str(w.message)
        ]
        self.assertEqual(len(prompt_mode_warnings), 1)

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

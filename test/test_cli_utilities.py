import contextlib
import io
import json
import unittest
from unittest.mock import patch

from rank_llm.cli.main import main


class TestCLIUtilities(unittest.TestCase):
    def test_evaluate_command_emits_summary_envelope(self):
        stdout = io.StringIO()
        with (
            patch(
                "rank_llm.cli.main.run_evaluate_aggregate",
                return_value={
                    "output_file": "trec_eval_aggregated_results_model.jsonl"
                },
            ) as mocked,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "evaluate",
                    "--model-name",
                    "model",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked.call_args.kwargs["model_name"], "model")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "evaluate")
        self.assertEqual(
            payload["artifacts"][0]["value"]["output_file"],
            "trec_eval_aggregated_results_model.jsonl",
        )

    def test_analyze_command_emits_summary_envelope(self):
        stdout = io.StringIO()
        with (
            patch(
                "rank_llm.cli.main.run_response_analysis_files",
                return_value={"files": ["one.json"], "metrics": {"errors": 0}},
            ) as mocked,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "analyze",
                    "--files",
                    "one.json",
                    "two.json",
                    "--verbose",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked.call_args.kwargs["files"], ["one.json", "two.json"])
        self.assertTrue(mocked.call_args.kwargs["verbose"])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "analyze")
        self.assertEqual(payload["artifacts"][0]["value"]["metrics"]["errors"], 0)

    def test_retrieve_cache_command_emits_summary_envelope(self):
        stdout = io.StringIO()
        with (
            patch(
                "rank_llm.cli.main.run_retrieve_cache_generation",
                return_value={"output_file": "cache.json", "record_count": 2},
            ) as mocked,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "retrieve-cache",
                    "--trec-file",
                    "run.trec",
                    "--collection-file",
                    "collection.tsv",
                    "--query-file",
                    "queries.tsv",
                    "--output-file",
                    "cache.json",
                    "--topk",
                    "10",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked.call_args.kwargs["trec_file"], "run.trec")
        self.assertEqual(mocked.call_args.kwargs["topk"], 10)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "retrieve-cache")
        self.assertEqual(payload["artifacts"][0]["value"]["record_count"], 2)


if __name__ == "__main__":
    unittest.main()

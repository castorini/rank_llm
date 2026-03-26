import contextlib
import io
import json
import unittest
from unittest.mock import patch

from rank_llm.cli import operations
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

    def test_evaluate_json_output_suppresses_runner_stdout(self):
        stdout = io.StringIO()

        def noisy_runner(model_name, context_size, rerank_results_dirname):
            print("diagnostic output")
            return None

        with (
            patch(
                "rank_llm.cli.main.run_evaluate_aggregate",
                side_effect=lambda **kwargs: operations.run_evaluate_aggregate(
                    runner=noisy_runner, **kwargs
                ),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--output", "json", "evaluate", "--model-name", "model"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "evaluate")
        self.assertNotIn("diagnostic output", stdout.getvalue())

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

    def test_analyze_json_output_suppresses_verbose_stdout(self):
        stdout = io.StringIO()

        def noisy_runner(files, verbose):
            print("bad record")
            return {"files": files, "metrics": {"errors": 1}}

        with (
            patch(
                "rank_llm.cli.main.run_response_analysis_files",
                side_effect=lambda **kwargs: operations.run_response_analysis_files(
                    runner=noisy_runner, **kwargs
                ),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "analyze",
                    "--files",
                    "one.json",
                    "--verbose",
                ]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "analyze")
        self.assertNotIn("bad record", stdout.getvalue())

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

    def test_retrieve_cache_json_output_suppresses_generator_stdout(self):
        stdout = io.StringIO()

        def noisy_generator(*args):
            print("loaded queries")
            return [{"query": "cats"}]

        def noisy_writer(output_file, results):
            print(f"wrote {output_file}")

        with (
            patch(
                "rank_llm.cli.main.run_retrieve_cache_generation",
                side_effect=lambda **kwargs: operations.run_retrieve_cache_generation(
                    generator=noisy_generator,
                    writer=noisy_writer,
                    **kwargs,
                ),
            ),
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
                ]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "retrieve-cache")
        self.assertNotIn("loaded queries", stdout.getvalue())
        self.assertNotIn("wrote cache.json", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

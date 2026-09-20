import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rank_llm.api.cli.main import main


class TestCLIUtilities(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(contextlib.chdir(directory))
        self.enterContext(
            patch("rank_llm.api.cli.main.load_config", return_value=({}, None))
        )

    def run_cli(self, *args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--output", "json", *args])
        self.assertEqual(code, 0, stdout.getvalue() + stderr.getvalue())
        return json.loads(stdout.getvalue())["artifacts"][0]["value"]

    def test_evaluate_aggregates_run_files(self):
        directory = Path("runs/BM25")
        directory.mkdir(parents=True)
        run = directory / "model_2048_20_dl19.trec"
        run.write_text("q1 Q0 d1 1 1.0 rank_llm\n")
        # Only the external trec_eval invocation is mocked.
        with patch(
            "rank_llm.scripts.run_trec_eval.EvalFunction.eval",
            return_value="metric 1.0",
        ) as evaluate:
            summary = self.run_cli(
                "evaluate",
                "--model-name",
                "model",
                "--context-size",
                "2048",
                "--rerank-results-dirname",
                "runs",
            )
        result = json.loads(Path(summary["output_file"]).read_text())
        self.assertEqual(result["file"], str(run))
        self.assertEqual(len(result["result"]), 4)
        self.assertEqual(evaluate.call_args.args[0][-1], str(run))

    def test_analyze_reads_invocation_history(self):
        Path("history.json").write_text(
            json.dumps(
                [
                    {
                        "invocations_history": [
                            {
                                "prompt": "[1] passage",
                                "response": "[1]",
                                "output_validation_regex": repr(r"\[\d+\]"),
                                "output_extraction_regex": repr(r"\[(\d+)\]"),
                            }
                        ]
                    }
                ]
            )
        )
        summary = self.run_cli("analyze", "--files", "history.json", "--verbose")
        self.assertEqual(
            summary["metrics"],
            {"ok": 1, "wrong_format": 0, "repetition": 0, "missing_documents": 0},
        )

    def test_retrieve_cache_reads_local_files_and_writes_outputs(self):
        Path("queries.tsv").write_text("q1\tcats\n")
        Path("collection.tsv").write_text("d1\tfirst passage\nd2\tsecond passage\n")
        Path("run.trec").write_text(
            "q1 Q0 d1 1 2.0 rank_llm\nq1 Q0 d2 2 1.0 rank_llm\n"
        )
        summary = self.run_cli(
            "retrieve-cache",
            "--trec-file",
            "run.trec",
            "--collection-file",
            "collection.tsv",
            "--query-file",
            "queries.tsv",
            "--output-file",
            "cache.json",
            "--output-trec-file",
            "cache.trec",
            "--topk",
            "1",
        )
        results = json.loads(Path("cache.json").read_text())
        self.assertEqual(summary["record_count"], 1)
        self.assertEqual(results[0]["query"], "cats")
        self.assertEqual([h["content"] for h in results[0]["hits"]], ["first passage"])
        self.assertEqual(Path("cache.trec").read_text(), "q1 Q0 d1 1 2.0 run_id\n")


if __name__ == "__main__":
    unittest.main()

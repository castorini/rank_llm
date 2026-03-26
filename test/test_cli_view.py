import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rank_llm.cli.main import main


class TestCLIView(unittest.TestCase):
    def test_view_detects_rerank_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            path.write_text(
                '{"query":{"text":"cats","qid":"q1"},"candidates":[{"docid":"d1","score":1.0,"doc":{"contents":"doc"}}]}\n',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--output", "json", "view", str(path)])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        summary = payload["artifacts"][0]["value"]
        self.assertEqual(summary["artifact_type"], "rerank-output")

    def test_view_detects_request_input_when_candidates_are_not_ranked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requests.jsonl"
            path.write_text(
                '{"query":{"text":"cats","qid":"q1"},"candidates":[{"text":"doc"}]}\n',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--output", "json", "view", str(path)])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        summary = payload["artifacts"][0]["value"]
        self.assertEqual(summary["artifact_type"], "request-input")

    def test_view_detects_trec_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.trec"
            path.write_text("q1 Q0 d1 1 1.0 rank_llm\n", encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--output", "json", "view", str(path)])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            payload["artifacts"][0]["value"]["artifact_type"], "trec-output"
        )

    def test_view_detects_invocations_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "query": {"text": "cats", "qid": "q1"},
                            "invocations_history": [{"prompt": "p", "response": "r"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--output", "json", "view", str(path)])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            payload["artifacts"][0]["value"]["artifact_type"],
            "invocations-history",
        )

    def test_view_renders_text_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            path.write_text(
                '{"query":"cats","candidates":[{"docid":"d1","score":1.0,"doc":{"contents":"doc"}}]}\n',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["view", str(path)])
        self.assertEqual(exit_code, 0)
        self.assertIn("type: rerank-output", stdout.getvalue())

    def test_view_fails_on_missing_file(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--output", "json", "view", "/tmp/does-not-exist.jsonl"])
        self.assertEqual(exit_code, 5)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")

    def test_view_invalid_json_returns_validation_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.jsonl"
            path.write_text('{"query":', encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--output", "json", "view", str(path)])
        self.assertEqual(exit_code, 5)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")
        self.assertIn("invalid JSON content", payload["errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()

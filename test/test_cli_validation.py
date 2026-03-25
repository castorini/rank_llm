import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rank_llm.cli.main import main


class TestCLIValidation(unittest.TestCase):
    def test_validate_rerank_accepts_valid_direct_payload(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "validate",
                    "rerank",
                    "--input-json",
                    '{"query":"cats","candidates":["doc"]}',
                ]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["validation"]["valid"])

    def test_validate_rerank_rejects_invalid_direct_payload(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "validate",
                    "rerank",
                    "--input-json",
                    '{"query":"cats"}',
                ]
            )
        self.assertEqual(exit_code, 5)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")

    def test_validate_rerank_accepts_valid_batch_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requests.jsonl"
            path.write_text(
                '{"query":"cats","candidates":["doc"]}\n',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output",
                        "json",
                        "validate",
                        "rerank",
                        "--requests-file",
                        str(path),
                    ]
                )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["validation"]["record_count"], 1)

    def test_validate_rerank_rejects_invalid_batch_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requests.jsonl"
            path.write_text('{"query":"cats"}\n', encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output",
                        "json",
                        "validate",
                        "rerank",
                        "--requests-file",
                        str(path),
                    ]
                )
        self.assertEqual(exit_code, 5)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["validation"]["valid"])

    def test_dry_run_does_not_execute_reranker(self):
        with patch("rank_llm.cli.main.run_mcp_retrieve_and_rerank") as mocked:
            exit_code = main(
                [
                    "rerank",
                    "--model-path",
                    "model",
                    "--dataset",
                    "dl19",
                    "--retrieval-method",
                    "bm25",
                    "--dry-run",
                ]
            )
        self.assertEqual(exit_code, 0)
        mocked.assert_not_called()

    def test_validate_only_returns_without_execution(self):
        with patch("rank_llm.cli.main.run_mcp_rerank") as mocked:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--output",
                        "json",
                        "rerank",
                        "--model-path",
                        "model",
                        "--input-json",
                        '{"query":"cats","candidates":["doc"]}',
                        "--validate-only",
                    ]
                )
        self.assertEqual(exit_code, 0)
        mocked.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["mode"], "validate")


if __name__ == "__main__":
    unittest.main()

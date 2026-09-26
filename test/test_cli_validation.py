import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rank_llm.cli.main import main


class TestCLIValidation(unittest.TestCase):
    def test_all_direct_validation_modes_reject_malformed_candidates(self):
        for command in (
            ["validate", "rerank"],
            ["rerank", "--model-path", "rank_random", "--validate-only"],
            ["rerank", "--model-path", "rank_random", "--dry-run"],
            ["rerank", "--model-path", "rank_random"],
        ):
            with self.subTest(command=command):
                stdout = io.StringIO()
                with (
                    patch("rank_llm.cli.main.run_mcp_rerank") as mocked,
                    contextlib.redirect_stdout(stdout),
                ):
                    exit_code = main(
                        [
                            "--output",
                            "json",
                            *command,
                            "--input-json",
                            '{"query":"q","candidates":5}',
                        ]
                    )
                self.assertEqual(exit_code, 5)
                self.assertEqual(
                    json.loads(stdout.getvalue())["status"], "validation_error"
                )
                mocked.assert_not_called()

    def test_validate_rerank_checks_nested_direct_input(self):
        invalid_payloads = (
            [],
            {"query": {"qid": "q1"}, "candidates": []},
            {"query": "q", "candidates": [42]},
            {"query": "q", "candidates": [{}]},
            {"query": "q", "candidates": [{"text": 42}]},
            {"query": "q", "candidates": [{"doc": "passage", "score": "high"}]},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "--output",
                            "json",
                            "validate",
                            "rerank",
                            "--input-json",
                            json.dumps(payload),
                        ]
                    )
                self.assertEqual(exit_code, 5)
                self.assertFalse(json.loads(stdout.getvalue())["validation"]["valid"])

    def test_validate_rerank_accepts_supported_candidate_forms(self):
        payload = {
            "query": {"text": "cats", "qid": 1},
            "candidates": [
                "plain text",
                {"text": "text field", "docid": "a", "score": 0.5},
                {"doc": "document field"},
                {"doc": {"contents": "document object"}},
            ],
        }
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "validate",
                    "rerank",
                    "--input-json",
                    json.dumps(payload),
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertTrue(json.loads(stdout.getvalue())["validation"]["valid"])

        stdout = io.StringIO()
        with (
            patch("rank_llm.cli.main.run_mcp_rerank", return_value=[]) as rerank,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "rerank",
                    "--model-path",
                    "rank_random",
                    "--input-json",
                    json.dumps(payload),
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(rerank.call_args.kwargs["query_text"], "cats")
        self.assertEqual(rerank.call_args.kwargs["query_id"], 1)
        self.assertEqual(len(rerank.call_args.kwargs["candidates"]), 4)

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

    def test_validate_rerank_rejects_malformed_batch_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requests.jsonl"
            path.write_text('{"query":"cats"\n', encoding="utf-8")
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
        self.assertIn("invalid JSON on line 1", payload["errors"][0]["message"])

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

    def test_dry_run_rejects_invalid_dataset_arguments(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "rerank",
                    "--model-path",
                    "model",
                    "--dataset",
                    "dl19",
                    "--dry-run",
                ]
            )
        self.assertEqual(exit_code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")
        self.assertIn("--retrieval-method is required", payload["errors"][0]["message"])

    def test_validate_only_rejects_requests_file_with_retrieval_method(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "rerank",
                    "--model-path",
                    "model",
                    "--requests-file",
                    "requests.jsonl",
                    "--retrieval-method",
                    "bm25",
                    "--validate-only",
                ]
            )
        self.assertEqual(exit_code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")
        self.assertIn(
            "--retrieval-method must not be used",
            payload["errors"][0]["message"],
        )


if __name__ == "__main__":
    unittest.main()

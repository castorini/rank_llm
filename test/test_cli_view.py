import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rank_llm.cli.main import main
from rank_llm.data import Candidate, DataWriter, Query, Result


class TestCLIView(unittest.TestCase):
    def test_view_reads_json_results_written_by_data_writer(self):
        results = [
            Result(
                query=Query(text="cats", qid=qid),
                candidates=[Candidate(docid="d1", score=1.0, doc={"contents": "doc"})],
            )
            for qid in ("q1", "q2")
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.json"
            DataWriter(results).write_in_json_format(str(path))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--output", "json", "view", str(path)])
        self.assertEqual(exit_code, 0)
        summary = json.loads(stdout.getvalue())["artifacts"][0]["value"]
        self.assertEqual(summary["artifact_type"], "rerank-output")
        self.assertEqual(summary["summary"], {"record_count": 2, "candidate_total": 2})
        self.assertEqual(len(summary["sampled_records"]), 1)
        self.assertEqual(summary["sampled_records"][0]["query"]["qid"], "q1")

    def test_view_detects_json_request_input(self):
        requests = [{"query": "猫咪 🐈", "candidates": [{"text": "中文文档 🐕"}]}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requests.json"
            path.write_text(json.dumps(requests, ensure_ascii=False), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--output", "json", "view", str(path)])
        self.assertEqual(exit_code, 0)
        summary = json.loads(stdout.getvalue())["artifacts"][0]["value"]
        self.assertEqual(summary["artifact_type"], "request-input")
        self.assertEqual(summary["sampled_records"], requests)

    def test_view_invalid_json_record_structure_is_validation_error(self):
        for contents in (
            "{}",
            "[42]",
            '[{"query":"cats","candidates":null}]',
            '[{"query":"cats","candidates":[]},42]',
            '[{"query":"cats","candidates":[]},{"query":"dogs","candidates":null}]',
        ):
            with (
                self.subTest(contents=contents),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                path = Path(temp_dir) / "unrecognized.json"
                path.write_text(contents, encoding="utf-8")
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["--output", "json", "view", str(path)])
                self.assertEqual(exit_code, 5)
                self.assertEqual(
                    json.loads(stdout.getvalue())["status"], "validation_error"
                )

    def test_view_empty_writer_outputs_show_zero_records(self):
        for method, extension, artifact_type, expected_summary in (
            ("write_in_json_format", ".json", "empty-json", {"record_count": 0}),
            ("write_in_jsonl_format", ".jsonl", "empty-jsonl", {"record_count": 0}),
            (
                "write_inference_invocations_history",
                ".json",
                "empty-json",
                {"record_count": 0},
            ),
            ("write_in_trec_eval_format", ".trec", "trec-output", {"line_count": 0}),
        ):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / f"empty{extension}"
                getattr(DataWriter([]), method)(str(path))
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["--output", "json", "view", str(path)])
                self.assertEqual(exit_code, 0)
                summary = json.loads(stdout.getvalue())["artifacts"][0]["value"]
                self.assertEqual(summary["artifact_type"], artifact_type)
                self.assertEqual(summary["summary"], expected_summary)
                self.assertEqual(summary["sampled_records"], [])

    def test_view_whitespace_only_jsonl_is_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.jsonl"
            path.write_text(" \n\t\n", encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--output", "json", "view", str(path)])
            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())["artifacts"][0]["value"]
            self.assertEqual(summary["artifact_type"], "empty-jsonl")
            self.assertEqual(summary["summary"], {"record_count": 0})

    def test_view_malformed_jsonl_and_history_return_validation_errors(self):
        for name, contents in (
            ("null.jsonl", "null\n"),
            ("scalar.jsonl", "42\n"),
            ("null-candidates.jsonl", '{"query":"cats","candidates":null}\n'),
            ("later-null.jsonl", '{"query":"cats","candidates":[]}\nnull\n'),
            (
                "later-bad-candidates.jsonl",
                '{"query":"cats","candidates":[]}\n{"query":"dogs","candidates":42}\n',
            ),
            ("bad-history.json", '[{"query":"cats","invocations_history":null}]'),
            (
                "later-bad-history.json",
                '[{"query":"cats","invocations_history":[]},42]',
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / name
                path.write_text(contents, encoding="utf-8")
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["--output", "json", "view", str(path)])
                self.assertEqual(exit_code, 5)
                self.assertEqual(
                    json.loads(stdout.getvalue())["status"], "validation_error"
                )

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

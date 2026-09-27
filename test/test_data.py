import json
import tempfile
import unittest
from pathlib import Path

from rank_llm.data import (
    Candidate,
    DataWriter,
    Query,
    Request,
    RerankValidationError,
    read_requests_from_file,
)


class TestRequestFiles(unittest.TestCase):
    def test_structured_requests_round_trip(self):
        requests = [
            Request(
                Query("café", "q1"),
                [
                    Candidate(
                        42,
                        -1.25,
                        {"contents": "passage", "title": "Title", "metadata": [1, 2]},
                    ),
                    Candidate("d2", 3.0, {"text": "second passage"}),
                ],
            ),
            Request(Query("empty", 2)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("json", "jsonl"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"requests.{suffix}"
                    writer = DataWriter(requests)
                    getattr(writer, f"write_in_{suffix}_format")(str(path))
                    self.assertEqual(read_requests_from_file(str(path)), requests)

    def test_text_candidates_and_optional_candidates(self):
        payloads = [
            {"query": "cats", "candidates": ["one", {"text": "two"}]},
            {"query": {"text": "empty", "qid": 2}},
        ]
        expected = [
            Request(
                Query("cats", ""),
                [
                    Candidate("1", 0.0, {"contents": "one"}),
                    Candidate("2", 0.0, {"contents": "two"}),
                ],
            ),
            Request(Query("empty", 2)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("json", "jsonl"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"requests.{suffix}"
                    path.write_text(
                        json.dumps(payloads)
                        if suffix == "json"
                        else "\n\n".join(json.dumps(p) for p in payloads) + "\n"
                    )
                    self.assertEqual(read_requests_from_file(str(path)), expected)

    def test_malformed_files_report_validation_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix, text, message in (
                ("json", "{}", "must contain an array"),
                ("json", "[", "invalid JSON"),
                ("jsonl", '\n{"query": "cats", "candidates": [123]}', "line 2"),
            ):
                with self.subTest(suffix=suffix, text=text):
                    path = Path(directory) / f"requests.{suffix}"
                    path.write_text(text)
                    with self.assertRaisesRegex(RerankValidationError, message):
                        read_requests_from_file(str(path))

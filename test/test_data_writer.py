import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rank_llm.data import Candidate, DataWriter, InferenceInvocation, Query, Result


def _result(qid: str) -> Result:
    return Result(
        query=Query(text=f"query {qid}", qid=qid),
        candidates=[Candidate(docid=f"doc {qid}", score=1.0, doc={"text": qid})],
        invocations_history=[
            InferenceInvocation(
                prompt=f"prompt {qid}",
                response=f"response {qid}",
                input_token_count=1,
                output_token_count=2,
            )
        ],
    )


class TestDataWriterAppend(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.filename = Path(self.directory.name) / "output.json"

    def test_result_json_append_merges_arrays_in_order(self):
        DataWriter(_result("first")).write_in_json_format(str(self.filename))
        DataWriter(_result("second"), append=True).write_in_json_format(
            str(self.filename)
        )

        records = json.loads(self.filename.read_text())
        self.assertEqual(
            [record["query"]["qid"] for record in records], ["first", "second"]
        )
        self.assertEqual(records[1]["candidates"][0]["docid"], "doc second")

    def test_invocation_history_append_merges_arrays_in_order(self):
        DataWriter(_result("first")).write_inference_invocations_history(
            str(self.filename)
        )
        DataWriter(_result("second"), append=True).write_inference_invocations_history(
            str(self.filename)
        )

        records = json.loads(self.filename.read_text())
        self.assertEqual(
            [record["query"]["qid"] for record in records], ["first", "second"]
        )
        self.assertEqual(
            records[1]["invocations_history"][0]["response"], "response second"
        )

    def test_append_creates_new_json_file(self):
        DataWriter(_result("first"), append=True).write_in_json_format(
            str(self.filename)
        )
        self.assertEqual(
            json.loads(self.filename.read_text())[0]["query"]["qid"], "first"
        )

    def test_append_treats_empty_existing_file_as_new_array(self):
        for contents in ("", " \n\t"):
            for write_method in (
                "write_in_json_format",
                "write_inference_invocations_history",
            ):
                with self.subTest(contents=contents, write_method=write_method):
                    self.filename.write_text(contents)
                    writer = DataWriter(_result("first"), append=True)
                    getattr(writer, write_method)(str(self.filename))
                    records = json.loads(self.filename.read_text())
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]["query"]["qid"], "first")

    def test_append_rejects_non_array_json_without_modifying_file(self):
        original = '{"query": "existing"}'
        self.filename.write_text(original)

        with self.assertRaisesRegex(ValueError, "Expected a JSON array"):
            DataWriter(_result("first"), append=True).write_in_json_format(
                str(self.filename)
            )

        self.assertEqual(self.filename.read_text(), original)

    def test_non_append_replaces_existing_json(self):
        DataWriter(_result("first")).write_in_json_format(str(self.filename))
        DataWriter(_result("second")).write_in_json_format(str(self.filename))
        self.assertEqual(
            [
                record["query"]["qid"]
                for record in json.loads(self.filename.read_text())
            ],
            ["second"],
        )


class TestDataWriterEncoding(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

        def non_utf8_open(*args, **kwargs):
            kwargs.setdefault("encoding", "cp1252")
            return open(*args, **kwargs)

        locale_open = patch(
            "rank_llm.data.open", side_effect=non_utf8_open, create=True
        )
        locale_open.start()
        self.addCleanup(locale_open.stop)

    def test_json_writers_preserve_unicode_when_appending(self):
        for method in (
            "write_in_json_format",
            "write_in_jsonl_format",
            "write_inference_invocations_history",
        ):
            with self.subTest(method=method):
                filename = Path(self.directory.name) / method
                first, second = _result("查询 😊"), _result("文档 📝")
                getattr(DataWriter(first), method)(str(filename))
                getattr(DataWriter(second, append=True), method)(str(filename))

                contents = filename.read_bytes().decode("utf-8")
                records = (
                    [json.loads(line) for line in contents.splitlines()]
                    if method == "write_in_jsonl_format"
                    else json.loads(contents)
                )
                self.assertEqual(
                    [record["query"]["text"] for record in records],
                    [first.query.text, second.query.text],
                )
                if method == "write_inference_invocations_history":
                    self.assertEqual(
                        records[1]["invocations_history"][0]["response"],
                        second.invocations_history[0].response,
                    )
                else:
                    self.assertEqual(
                        records[1]["candidates"][0]["doc"], second.candidates[0].doc
                    )

    def test_append_preserves_existing_utf8_array(self):
        for method in (
            "write_in_json_format",
            "write_inference_invocations_history",
        ):
            with self.subTest(method=method):
                filename = Path(self.directory.name) / method
                existing = {"query": {"text": "查询 😊", "qid": "first"}}
                filename.write_text(
                    json.dumps([existing], ensure_ascii=False), encoding="utf-8"
                )
                getattr(DataWriter(_result("second"), append=True), method)(
                    str(filename)
                )
                records = json.loads(filename.read_bytes().decode("utf-8"))
                self.assertEqual(records[0], existing)
                self.assertEqual(records[1]["query"]["qid"], "second")

    def test_trec_writer_preserves_unicode_when_appending(self):
        filename = Path(self.directory.name) / "output.txt"
        first, second = _result("查询😊"), _result("文档📝")
        first.candidates[0].docid = "文档😊"
        second.candidates[0].docid = "文档📝"
        DataWriter(first).write_in_trec_eval_format(str(filename))
        DataWriter(second, append=True).write_in_trec_eval_format(str(filename))
        self.assertEqual(
            filename.read_bytes().decode("utf-8").splitlines(),
            [
                f"{first.query.qid} Q0 {first.candidates[0].docid} 1 1.0 rank_llm",
                f"{second.query.qid} Q0 {second.candidates[0].docid} 1 1.0 rank_llm",
            ],
        )


if __name__ == "__main__":
    unittest.main()

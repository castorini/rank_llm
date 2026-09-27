import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

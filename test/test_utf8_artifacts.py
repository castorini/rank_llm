import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import (
    Candidate,
    DataWriter,
    InferenceInvocation,
    Query,
    Result,
    read_requests_from_file,
)
from rank_llm.retrieve.retrieval_method import RetrievalMethod
from rank_llm.retrieve.retriever import RetrievalMode, Retriever

_OPEN = open


def _locale_open(encoding):
    def open_with_locale(file, mode="r", *args, **kwargs):
        if "b" not in mode and kwargs.get("encoding") is None:
            kwargs["encoding"] = encoding
        return _OPEN(file, mode, *args, **kwargs)

    return patch("builtins.open", open_with_locale)


def _result(index):
    return Result(
        query=Query(text=f"中文查询 {index} 🐈", qid=f"查询{index}"),
        candidates=[
            Candidate(
                docid=f"文档{index}", score=1.0, doc={"contents": "中文 résumé 🚀"}
            )
        ],
        invocations_history=[
            InferenceInvocation(
                prompt="中文资料 🧪 [1] 文档 [2] 文档",
                response="中文解释 🐈 [1] > [2]",
                input_token_count=10,
                output_token_count=5,
                output_validation_regex=repr(r"^\[\d+\]( > \[\d+\])*$"),
                output_extraction_regex=repr(r"\[(\d+)\]"),
            )
        ],
    )


def _request_record(result):
    return {
        "query": asdict(result.query),
        "candidates": [asdict(candidate) for candidate in result.candidates],
    }


class TestUTF8Artifacts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_write_and_append_unicode_artifacts(self):
        for encoding in ("cp1252", "gbk"):
            for method, extension in (
                ("write_in_json_format", ".json"),
                ("write_in_jsonl_format", ".jsonl"),
                ("write_inference_invocations_history", ".json"),
                ("write_in_trec_eval_format", ".trec"),
            ):
                with self.subTest(encoding=encoding, method=method):
                    path = self.root / f"{encoding}-{method}{extension}"
                    with _locale_open(encoding):
                        getattr(DataWriter(_result(1)), method)(str(path))
                        getattr(DataWriter(_result(2), append=True), method)(str(path))
                    text = path.read_text(encoding="utf-8")
                    if extension == ".trec":
                        self.assertEqual(
                            text.splitlines(),
                            [
                                "查询1 Q0 文档1 1 1.0 rank_llm",
                                "查询2 Q0 文档2 1 1.0 rank_llm",
                            ],
                        )
                    else:
                        records = (
                            [json.loads(line) for line in text.splitlines()]
                            if extension == ".jsonl"
                            else json.loads(text)
                        )
                        self.assertEqual(
                            [record["query"]["text"] for record in records],
                            [_result(1).query.text, _result(2).query.text],
                        )
                        if method == "write_inference_invocations_history":
                            self.assertEqual(
                                records[1]["invocations_history"][0]["prompt"],
                                _result(2).invocations_history[0].prompt,
                            )
                        else:
                            self.assertEqual(
                                records[1]["candidates"][0]["doc"],
                                _result(2).candidates[0].doc,
                            )

    def test_append_preserves_existing_utf8_records(self):
        for encoding in ("cp1252", "gbk"):
            for method, record in (
                ("write_in_json_format", _request_record(_result(1))),
                (
                    "write_inference_invocations_history",
                    {
                        "query": asdict(_result(1).query),
                        "invocations_history": [
                            asdict(_result(1).invocations_history[0])
                        ],
                    },
                ),
            ):
                with self.subTest(encoding=encoding, method=method):
                    path = self.root / "existing.json"
                    path.write_text(
                        json.dumps([record], ensure_ascii=False), encoding="utf-8"
                    )
                    with _locale_open(encoding):
                        getattr(DataWriter(_result(2), append=True), method)(str(path))
                    records = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(records[0], record)
                    self.assertEqual(records[1]["query"], asdict(_result(2).query))

    def test_public_request_reader_uses_utf8(self):
        record = _request_record(_result(1))
        for encoding in ("cp1252", "gbk"):
            for extension in ("json", "jsonl"):
                with self.subTest(encoding=encoding, extension=extension):
                    path = self.root / f"requests.{extension}"
                    data = [record] if extension == "json" else record
                    path.write_text(
                        json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                    with _locale_open(encoding):
                        requests = read_requests_from_file(str(path))
                    self.assertEqual(requests[0].query, _result(1).query)
                    self.assertEqual(requests[0].candidates, _result(1).candidates)

    def test_saved_response_analysis_uses_utf8(self):
        result = _result(1)
        record = {
            "query": asdict(result.query),
            "invocations_history": [asdict(result.invocations_history[0])],
        }
        path = self.root / "history.json"
        path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
        for encoding in ("cp1252", "gbk"):
            with self.subTest(encoding=encoding), _locale_open(encoding):
                responses, passages, _, _ = ResponseAnalyzer.from_stored_files(
                    [str(path)]
                ).read_saved_responses()
                self.assertEqual(responses, [result.invocations_history[0].response])
                self.assertEqual(passages, [2])

    def test_cached_retrieval_uses_utf8(self):
        record = _request_record(_result(1))
        jsonl = self.root / "cached.jsonl"
        jsonl.write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        custom = self.root / "CUSTOM_INDEX" / "retrieve_results_test_top1.json"
        custom.parent.mkdir()
        custom.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
        for encoding in ("cp1252", "gbk"):
            for route in ("local", "downloaded", "custom"):
                with self.subTest(encoding=encoding, route=route):
                    retriever = Retriever(
                        RetrievalMode.CUSTOM
                        if route == "custom"
                        else RetrievalMode.DATASET,
                        dataset="test",
                        retrieval_method=RetrievalMethod.CUSTOM_INDEX
                        if route == "custom"
                        else RetrievalMethod.BM25,
                    )
                    with (
                        _locale_open(encoding),
                        patch(
                            "rank_llm.retrieve.retriever.get_cache_home",
                            return_value=str(self.root),
                        ),
                        patch.object(
                            Retriever,
                            "_get_file_with_highest_k",
                            return_value=(str(jsonl), 1)
                            if route == "local"
                            else (None, -1),
                        ),
                        patch(
                            "rank_llm.retrieve.retriever.download_cached_hits",
                            return_value=str(jsonl),
                        ) as download,
                    ):
                        requests = retriever.retrieve(k=1)
                    self.assertEqual(requests[0].query, _result(1).query)
                    self.assertEqual(requests[0].candidates, _result(1).candidates)
                    if route != "downloaded":
                        download.assert_not_called()

    def test_failed_append_does_not_modify_existing_file(self):
        for encoding in ("cp1252", "gbk"):
            with self.subTest(encoding=encoding):
                path = self.root / "broken.json"
                original = b'{"existing":'
                path.write_bytes(original)
                with _locale_open(encoding), self.assertRaises(json.JSONDecodeError):
                    DataWriter(_result(1), append=True).write_in_json_format(str(path))
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

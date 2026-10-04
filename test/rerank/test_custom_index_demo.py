import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.demo import rerank_dataset_with_custom_index as demo


class CustomIndexDemoTest(unittest.TestCase):
    def test_k_defaults_to_100(self):
        self.assertEqual(demo._build_parser().parse_args([]).k, 100)
        request = Request(query=Query(text="query", qid="1"))
        with (
            patch.object(demo, "_prepare_data"),
            patch.object(
                demo.Retriever, "from_custom_index", return_value=[request]
            ) as retrieve,
            patch.object(demo, "VicunaReranker") as model,
            patch.object(demo, "DataWriter"),
            tempfile.TemporaryDirectory() as output,
        ):
            demo.main(["--output-dir", output])
            self.assertEqual(retrieve.call_args.kwargs["k"], 100)
            self.assertEqual(
                model.return_value.rerank_batch.call_args.kwargs["rank_end"], 100
            )
            model.return_value.close.assert_called_once()

    def test_limits_queries_candidates_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path = root / "index"
            topics_path = root / "topics.tsv"
            index_path.mkdir()
            topics_path.write_text("topics")
            requests = [
                Request(
                    query=Query(text=f"query {i}", qid=str(i)),
                    candidates=[
                        Candidate(docid="a", score=1.0, doc={"segment": "a"}),
                        Candidate(docid="b", score=0.5, doc={"segment": "b"}),
                    ],
                )
                for i in range(2)
            ]

            class FakeReranker:
                def __init__(self, **kwargs):
                    pass

                def rerank_batch(self, requests, **kwargs):
                    self_test.assertEqual(len(requests), 1)
                    self_test.assertEqual(kwargs["rank_end"], 1)
                    self_test.assertEqual(kwargs["top_k_retrieve"], 1)
                    self.requests = requests
                    self.kwargs = kwargs
                    return [
                        Result(query=request.query, candidates=request.candidates)
                        for request in requests
                    ]

                def close(self):
                    pass

            def retrieve_with_k(**kwargs):
                return [
                    Request(
                        query=request.query,
                        candidates=request.candidates[: kwargs["k"]],
                    )
                    for request in requests
                ]

            output_dir = root / "output"
            self_test = self
            with (
                patch.object(demo, "VicunaReranker", FakeReranker),
                patch.object(
                    demo.Retriever,
                    "from_custom_index",
                    side_effect=retrieve_with_k,
                ) as retrieve,
            ):
                demo.main(
                    [
                        "--index-path",
                        str(index_path),
                        "--topics-path",
                        str(topics_path),
                        "--k",
                        "1",
                        "--num-queries",
                        "1",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
                history = output_dir / "rank_vicuna_7b_v1" / "invocations.json"
                self.assertTrue(history.exists())
                demo.main(
                    [
                        "--index-path",
                        str(index_path),
                        "--topics-path",
                        str(topics_path),
                        "--k",
                        "1",
                        "--num-queries",
                        "1",
                        "--output-dir",
                        str(output_dir),
                        "--no-populate-invocations-history",
                    ]
                )
                self.assertFalse(history.exists())

            self.assertEqual(retrieve.call_count, 2)
            self.assertEqual(retrieve.call_args.kwargs["k"], 1)
            folder = output_dir / "rank_vicuna_7b_v1"
            self.assertTrue((folder / "rerank.jsonl").is_file())
            self.assertTrue((folder / "rerank.txt").is_file())
            row = json.loads((folder / "rerank.jsonl").read_text().strip())
            self.assertEqual(row["query"]["qid"], "0")
            self.assertEqual(len(row["candidates"]), 1)

    def test_positive_numeric_arguments_are_validated(self):
        for option in (
            "--k",
            "--batch-size",
            "--context-size",
            "--window-size",
            "--stride",
            "--num-gpus",
            "--num-queries",
        ):
            for value in ("0", "-1"):
                with (
                    self.subTest(option=option, value=value),
                    patch.object(demo, "_prepare_data") as prepare,
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    demo.main([option, value])
                self.assertEqual(error.exception.code, 2)
                prepare.assert_not_called()

    def test_missing_custom_paths_do_not_download_default_data(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(demo, "urlretrieve") as download,
        ):
            args = demo._build_parser().parse_args(
                [
                    "--index-path",
                    str(Path(temp) / "custom"),
                    "--topics-path",
                    str(Path(temp) / "topics.tsv"),
                ]
            )
            with self.assertRaisesRegex(ValueError, "Missing custom index"):
                demo._prepare_data(args)
            download.assert_not_called()

    def test_download_extract_and_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "segments").write_text("index fixture")
            archive_path = root / "index.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(source, arcname="custom")
            topics = root / "topics.tsv"
            topics.write_text("1\tquery\n")
            args = demo._build_parser().parse_args(
                [
                    "--index-path",
                    str(root / "download" / "custom"),
                    "--topics-path",
                    str(root / "download" / "topics.tsv"),
                    "--index-url",
                    archive_path.as_uri(),
                    "--topics-url",
                    topics.as_uri(),
                ]
            )
            demo._prepare_data(args)
            self.assertEqual(
                (Path(args.index_path) / "segments").read_text(), "index fixture"
            )
            self.assertEqual(Path(args.topics_path).read_text(), "1\tquery\n")
            with patch.object(demo, "urlretrieve") as download:
                demo._prepare_data(args)
                download.assert_not_called()

    def test_close_on_inference_failure(self):
        request = Request(query=Query(text="query", qid="1"))
        with (
            patch.object(demo, "_prepare_data"),
            patch.object(demo.Retriever, "from_custom_index", return_value=[request]),
            patch.object(demo, "VicunaReranker") as model,
        ):
            model.return_value.rerank_batch.side_effect = RuntimeError(
                "inference failed"
            )
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                demo.main([])
            model.return_value.close.assert_called_once()

    def test_wrong_archive_layout_does_not_create_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "wrong.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                entry = tarfile.TarInfo("different")
                entry.type = tarfile.DIRTYPE
                archive.addfile(entry)
            topics = root / "topics.tsv"
            topics.write_text("1\tquery\n")
            args = demo._build_parser().parse_args(
                [
                    "--index-path",
                    str(root / "custom"),
                    "--topics-path",
                    str(topics),
                    "--index-url",
                    archive_path.as_uri(),
                ]
            )
            with self.assertRaisesRegex(ValueError, "must contain a directory"):
                demo._prepare_data(args)
            self.assertFalse(Path(args.index_path).exists())

    def test_stride_larger_than_window_fails_before_download(self):
        with (
            patch.object(demo, "_prepare_data") as prepare,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            demo.main(["--window-size", "2", "--stride", "3"])
        self.assertEqual(error.exception.code, 2)
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()

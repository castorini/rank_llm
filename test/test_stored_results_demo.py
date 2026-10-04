import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rank_llm.data import Result
from rank_llm.demo import rerank_stored_retrieved_results as demo


class StoredResultsDemoTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_file = self.root / "requests.jsonl"
        self.records = [
            {
                "query": {"text": "query", "qid": str(i)},
                "candidates": [
                    {"docid": str(j), "score": 1.0, "doc": {"text": str(j)}}
                    for j in range(count)
                ],
            }
            for i, count in enumerate((120, 3, 140))
        ]
        self.input_file.write_text(
            "\n".join(json.dumps(record) for record in self.records)
        )
        self.model = self.enterContext(patch.object(demo, "ZephyrReranker"))
        self.model.return_value.rerank_batch.side_effect = lambda requests, **kw: [
            Result(query=request.query, candidates=request.candidates)
            for request in requests
        ]
        self.enterContext(redirect_stdout(io.StringIO()))

    def run_demo(self, *args):
        demo.main(
            [
                "--input-file",
                str(self.input_file),
                "--output-dir",
                str(self.root / "output"),
                *args,
            ]
        )
        return self.model.return_value.rerank_batch.call_args.kwargs

    def test_omitted_k_covers_all_candidates_in_all_queries(self):
        call = self.run_demo()
        self.assertEqual(call["rank_end"], 140)
        self.assertEqual(call["top_k_retrieve"], 140)
        self.assertEqual([len(r.candidates) for r in call["requests"]], [120, 3, 140])
        self.model.return_value.close.assert_called_once()

    def test_query_limit_applies_before_default_depth_is_calculated(self):
        call = self.run_demo("--num-queries", "2")
        self.assertEqual([r.query.qid for r in call["requests"]], ["0", "1"])
        self.assertEqual(call["rank_end"], 120)

    def test_explicit_k_truncates_candidates_and_preserves_short_queries(self):
        call = self.run_demo("--num-queries", "2", "--k", "4")
        self.assertEqual(call["rank_end"], 4)
        self.assertEqual(call["top_k_retrieve"], 4)
        self.assertEqual(
            [[c.docid for c in r.candidates] for r in call["requests"]],
            [["0", "1", "2", "3"], ["0", "1", "2"]],
        )
        folder = self.root / "output" / "rank_zephyr_7b_v1_full"
        rows = [
            json.loads(line)
            for line in (folder / "rerank.jsonl").read_text().splitlines()
        ]
        self.assertEqual([len(row["candidates"]) for row in rows], [4, 3])

    def test_nonpositive_numeric_arguments_fail_before_loading_input(self):
        for option in (
            "--num-queries",
            "--k",
            "--batch-size",
            "--context-size",
            "--window-size",
            "--stride",
            "--num-gpus",
        ):
            for value in ("0", "-1"):
                with (
                    self.subTest(option=option, value=value),
                    patch.object(demo, "read_requests_from_file") as read,
                    redirect_stderr(io.StringIO()) as stderr,
                    self.assertRaises(SystemExit) as error,
                ):
                    self.run_demo(option, value)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(f"{option} must be greater than 0", stderr.getvalue())
                read.assert_not_called()
        self.model.assert_not_called()

    def test_stride_cannot_exceed_window(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.run_demo("--window-size", "2", "--stride", "3")
        self.assertEqual(error.exception.code, 2)
        self.model.assert_not_called()


if __name__ == "__main__":
    unittest.main()

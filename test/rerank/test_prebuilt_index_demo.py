import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.demo import rerank_dataset_with_prebuilt_index as demo
from rank_llm.retrieve import RetrievalMethod


class PrebuiltIndexDemoTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        def retrieve(dataset, method, k):
            return [
                Request(
                    query=Query(text="query", qid=f"{method.value}-{i}"),
                    candidates=[
                        Candidate(docid=str(j), score=1.0, doc={"text": str(j)})
                        for j in range(min(k, 6))
                    ],
                )
                for i in range(3)
            ]

        self.retrieve = self.enterContext(
            patch.object(
                demo.Retriever, "from_dataset_with_prebuilt_index", side_effect=retrieve
            )
        )
        self.evaluate = self.enterContext(
            patch.object(demo.EvalFunction, "from_results", return_value="0.5")
        )
        self.analyzer = self.enterContext(patch.object(demo, "ResponseAnalyzer"))
        self.model = self.enterContext(patch.object(demo, "VicunaReranker"))
        self.model.return_value.rerank_batch.side_effect = lambda requests, **kw: [
            Result(query=request.query, candidates=request.candidates)
            for request in requests
        ]
        self.enterContext(redirect_stdout(io.StringIO()))

    def run_demo(self, *args):
        demo.main(["--output-dir", str(self.root / "output"), *args])

    def test_defaults_match_original_demo(self):
        args = demo._build_parser().parse_args([])
        self.assertEqual(args.dataset, "dl19")
        self.assertEqual(args.model, "castorini/rank_vicuna_7b_v1")
        self.assertEqual(args.k, 100)
        self.assertTrue(args.populate_invocations_history)

        self.run_demo()
        self.assertEqual(
            [call.args[:2] for call in self.retrieve.call_args_list],
            [
                ("dl19", RetrievalMethod.BM25),
                ("dl19", RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL),
            ],
        )
        self.model.assert_called_once_with(
            model_path="castorini/rank_vicuna_7b_v1",
            context_size=4096,
            window_size=20,
            stride=10,
            batch_size=32,
            num_gpus=1,
        )
        self.model.return_value.close.assert_called_once()
        kwargs = self.model.return_value.rerank_batch.call_args.kwargs
        self.assertEqual(kwargs["rank_end"], 100)
        self.assertEqual(kwargs["top_k_retrieve"], 100)
        # Three retrieval metrics and one rerank metric per method.
        self.assertEqual(self.evaluate.call_count, 8)
        self.assertEqual(self.analyzer.from_inline_results.call_count, 2)
        for method in demo.DEFAULT_RETRIEVAL_METHODS:
            folder = self.root / "output" / method / "rank_vicuna_7b_v1"
            for name in ("rerank.jsonl", "rerank.txt", "invocations.json"):
                self.assertTrue((folder / name).exists())

    def test_limits_queries_and_candidates(self):
        self.run_demo("--retrieval-methods", "bm25", "--num-queries", "1", "--k", "4")
        self.assertEqual(self.retrieve.call_args.kwargs["k"], 4)
        kwargs = self.model.return_value.rerank_batch.call_args.kwargs
        self.assertEqual(len(kwargs["requests"]), 1)
        self.assertEqual(kwargs["rank_end"], 4)
        rows = (
            (self.root / "output" / "bm25" / "rank_vicuna_7b_v1" / "rerank.jsonl")
            .read_text()
            .splitlines()
        )
        self.assertEqual([len(json.loads(row)["candidates"]) for row in rows], [4])

    def test_disabling_history_removes_stale_file(self):
        self.run_demo("--retrieval-methods", "bm25")
        history = (
            self.root / "output" / "bm25" / "rank_vicuna_7b_v1" / "invocations.json"
        )
        self.assertTrue(history.exists())
        self.run_demo(
            "--retrieval-methods", "bm25", "--no-populate-invocations-history"
        )
        self.assertFalse(history.exists())
        # Response analysis needs the invocations history, so it is skipped.
        self.assertEqual(self.analyzer.from_inline_results.call_count, 1)

    def test_reranker_closed_when_retrieval_fails(self):
        self.retrieve.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.run_demo()
        self.model.return_value.close.assert_called_once()

    def test_nonpositive_numeric_arguments_fail_before_loading(self):
        for option in (
            "--k",
            "--num-queries",
            "--batch-size",
            "--context-size",
            "--window-size",
            "--stride",
            "--num-gpus",
        ):
            for value in ("0", "-1"):
                with (
                    self.subTest(option=option, value=value),
                    redirect_stderr(io.StringIO()) as stderr,
                    self.assertRaises(SystemExit) as error,
                ):
                    self.run_demo(option, value)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(f"{option} must be greater than 0", stderr.getvalue())
        self.retrieve.assert_not_called()
        self.model.assert_not_called()

    def test_invalid_stride_dataset_and_method_fail(self):
        for args in (
            ("--window-size", "2", "--stride", "3"),
            ("--dataset", "nope"),
            ("--retrieval-methods", "custom_index"),
        ):
            with (
                self.subTest(args=args),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                self.run_demo(*args)
            self.assertEqual(error.exception.code, 2)
        self.retrieve.assert_not_called()
        self.model.assert_not_called()


if __name__ == "__main__":
    unittest.main()

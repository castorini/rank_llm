import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.demo import readme_snippet as demo
from rank_llm.retrieve import RetrievalMethod


class ReadmeSnippetDemoTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        def retrieve(dataset, method, k):
            return [
                Request(
                    query=Query(text="query", qid=str(i)),
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
        self.enterContext(
            patch.object(demo.EvalFunction, "from_results", return_value="0.5")
        )
        self.analyzer = self.enterContext(patch.object(demo, "ResponseAnalyzer"))
        self.zephyr = self.enterContext(patch.object(demo, "ZephyrReranker"))
        self.vicuna = self.enterContext(patch.object(demo, "VicunaReranker"))
        for model in (self.zephyr, self.vicuna):
            model.return_value.rerank_batch.side_effect = lambda requests, **kw: [
                Result(query=request.query, candidates=request.candidates)
                for request in requests
            ]
        self.enterContext(redirect_stdout(io.StringIO()))

    def run_demo(self, *args):
        demo.main(["--output-dir", str(self.root / "output"), *args])

    def test_defaults_match_original_demo(self):
        self.run_demo()
        self.assertEqual(self.retrieve.call_args.args, ("dl19", RetrievalMethod.BM25))
        self.assertEqual(self.retrieve.call_args.kwargs["k"], 100)
        self.zephyr.assert_called_once_with(
            model_path="castorini/rank_zephyr_7b_v1_full",
            context_size=4096,
            window_size=20,
            stride=10,
            batch_size=32,
            num_gpus=1,
        )
        self.vicuna.assert_not_called()
        self.zephyr.return_value.close.assert_called_once()
        kwargs = self.zephyr.return_value.rerank_batch.call_args.kwargs
        self.assertEqual(kwargs["rank_end"], 100)
        self.assertEqual(kwargs["top_k_retrieve"], 100)
        self.assertTrue(kwargs["populate_invocations_history"])
        self.analyzer.from_inline_results.assert_called_once()
        folder = self.root / "output" / "rank_zephyr_7b_v1_full"
        for name in ("rerank.jsonl", "rerank.txt", "invocations.json"):
            self.assertTrue((folder / name).exists())

    def test_vicuna_uses_its_default_model_unless_overridden(self):
        self.run_demo("--reranker", "vicuna")
        self.assertEqual(
            self.vicuna.call_args.kwargs["model_path"], "castorini/rank_vicuna_7b_v1"
        )
        self.run_demo("--reranker", "vicuna", "--model", "org/Custom-Model")
        self.assertEqual(self.vicuna.call_args.kwargs["model_path"], "org/Custom-Model")
        self.assertTrue(
            (self.root / "output" / "custom-model" / "rerank.jsonl").exists()
        )
        self.zephyr.assert_not_called()

    def test_limits_queries_and_candidates(self):
        self.run_demo(
            "--retrieval-method",
            "SPLADE++_EnsembleDistil_ONNX",
            "--num-queries",
            "1",
            "--k",
            "4",
        )
        self.assertEqual(
            self.retrieve.call_args.args[1], RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL
        )
        self.assertEqual(self.retrieve.call_args.kwargs["k"], 4)
        kwargs = self.zephyr.return_value.rerank_batch.call_args.kwargs
        self.assertEqual(len(kwargs["requests"]), 1)
        self.assertEqual(kwargs["rank_end"], 4)
        rows = (
            (self.root / "output" / "rank_zephyr_7b_v1_full" / "rerank.jsonl")
            .read_text()
            .splitlines()
        )
        self.assertEqual([len(json.loads(row)["candidates"]) for row in rows], [4])

    def test_disabling_history_removes_stale_file(self):
        self.run_demo()
        history = self.root / "output" / "rank_zephyr_7b_v1_full" / "invocations.json"
        self.assertTrue(history.exists())
        self.analyzer.from_inline_results.reset_mock()
        self.run_demo("--no-populate-invocations-history")
        self.assertFalse(history.exists())
        # Response analysis needs the invocations history, so it is skipped.
        self.analyzer.from_inline_results.assert_not_called()

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
        self.zephyr.assert_not_called()

    def test_invalid_stride_dataset_and_method_fail(self):
        for args in (
            ("--window-size", "2", "--stride", "3"),
            ("--dataset", "nope"),
            ("--retrieval-method", "custom_index"),
            ("--reranker", "gpt"),
        ):
            with (
                self.subTest(args=args),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                self.run_demo(*args)
            self.assertEqual(error.exception.code, 2)
        self.retrieve.assert_not_called()
        self.zephyr.assert_not_called()


if __name__ == "__main__":
    unittest.main()

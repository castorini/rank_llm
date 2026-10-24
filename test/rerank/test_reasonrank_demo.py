import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.demo import rerank_reasonrank_bm25_beir as demo


class ReasonRankDemoTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        def retrieve(dataset, k):
            return [
                Request(
                    query=Query(text="query", qid=f"{dataset}-{i}"),
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
        self.model = self.enterContext(patch.object(demo, "RankListwiseOSLLM"))
        self.reranker = self.enterContext(patch.object(demo, "Reranker"))
        self.reranker.return_value.rerank_batch.side_effect = lambda requests, **kw: [
            Result(query=request.query, candidates=request.candidates)
            for request in requests
        ]

    def run_demo(self, *args):
        demo.main(["--output-dir", str(self.root / "output"), *args])

    def test_defaults_match_original_demo(self):
        args = demo._build_parser().parse_args([])
        self.assertEqual(args.datasets, demo.DEFAULT_DATASETS)
        self.assertEqual(args.model, "liuwenhan/reasonrank-32B")
        self.assertEqual(args.k, 100)
        self.assertEqual(args.context_size, 32768)
        self.assertEqual(args.reasoning_token_budget, 3072)
        self.assertEqual((args.window_size, args.stride, args.num_gpus), (20, 10, 1))
        self.assertTrue(args.populate_invocations_history)

        self.run_demo()
        self.assertEqual(
            [call.args[0] for call in self.retrieve.call_args_list],
            demo.DEFAULT_DATASETS,
        )
        self.assertEqual(
            [call.kwargs["batch_size"] for call in self.model.call_args_list],
            [1, 1, 1, 32, 32, 32, 32],
        )
        self.assertEqual(self.model.return_value.close.call_count, 7)
        rerank_kwargs = self.reranker.return_value.rerank_batch.call_args.kwargs
        self.assertEqual(rerank_kwargs["rank_end"], 100)
        self.assertEqual(rerank_kwargs["top_k_retrieve"], 100)
        folder = self.root / "output" / "reasonrank-32b"
        for suffix in (".jsonl", ".txt", "_invocations.json", "_metrics.json"):
            self.assertTrue((folder / f"scifact_bm25_top100{suffix}").exists())

    def test_batch_size_overrides_per_dataset_rule(self):
        self.run_demo("--datasets", "scifact", "covid", "--batch-size", "4")
        self.assertEqual(
            [call.kwargs["batch_size"] for call in self.model.call_args_list], [4, 4]
        )

    def test_limits_queries_and_candidates(self):
        self.run_demo("--datasets", "scifact", "--num-queries", "1", "--k", "4")
        self.assertEqual(self.retrieve.call_args.kwargs["k"], 4)
        kwargs = self.reranker.return_value.rerank_batch.call_args.kwargs
        self.assertEqual(len(kwargs["requests"]), 1)
        self.assertEqual(kwargs["rank_end"], 4)
        rows = (
            (self.root / "output" / "reasonrank-32b" / "scifact_bm25_top4.jsonl")
            .read_text()
            .splitlines()
        )
        self.assertEqual([len(json.loads(row)["candidates"]) for row in rows], [4])

    def test_disabling_history_removes_stale_file(self):
        self.run_demo("--datasets", "scifact")
        history = (
            self.root
            / "output"
            / "reasonrank-32b"
            / "scifact_bm25_top100_invocations.json"
        )
        self.assertTrue(history.exists())
        self.run_demo("--datasets", "scifact", "--no-populate-invocations-history")
        self.assertFalse(history.exists())

    def test_nonpositive_numeric_arguments_fail_before_retrieval(self):
        for option in (
            "--k",
            "--num-queries",
            "--batch-size",
            "--context-size",
            "--reasoning-token-budget",
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

    def test_stride_cannot_exceed_window_and_datasets_must_exist(self):
        for args in (("--window-size", "2", "--stride", "3"), ("--datasets", "nope")):
            with (
                self.subTest(args=args),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                self.run_demo(*args)
            self.assertEqual(error.exception.code, 2)
        self.retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()

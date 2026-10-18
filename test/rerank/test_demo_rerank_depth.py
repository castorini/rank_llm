import importlib
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank.listwise.rank_gemini import SafeGenai
from rank_llm.rerank.listwise.rank_gpt import SafeOpenai


class DemoRerankDepthTest(unittest.TestCase):
    def test_demos_pass_requested_depth_to_sync_and_async_calls(self):
        cases = (
            ("rerank_lit5", ["--variant", "both"], {"rerank_batch": 2}),
            ("rerank_lit5", ["--variant", "distill"], {"rerank_batch": 1}),
            ("rerank_lit5", ["--variant", "score"], {"rerank_batch": 1}),
            ("rerank_rank_gpt", [], {"rerank_batch": 1}),
            ("rerank_qwen_local", [], {"rerank_batch": 1}),
            ("rerank_qwen_async", [], {"rerank_async": 1}),
            ("rerank_gemini", [], {"rerank_batch": 1}),
            ("rerank_pointwise_vllm", [], {"rerank_batch": 1, "rerank_batch_async": 1}),
            (
                "experimental_results",
                [
                    "--datasets",
                    "dl19",
                    "--rerankers",
                    "rank_gpt",
                    "--skip-analysis",
                    "--skip-eval",
                ],
                {"rerank_batch": 1},
            ),
        )
        request = Request(query=Query(text="query", qid="1"))
        for name, args, expected_calls in cases:
            demo = importlib.import_module(f"rank_llm.demo.{name}")
            for depth in (None, 10, 200):
                with self.subTest(demo=name, args=args, depth=depth):
                    model = Mock()
                    model.rerank_batch.return_value = [request]
                    model.rerank_batch_async = AsyncMock(return_value=[request])
                    model.rerank_async = AsyncMock(return_value=request)
                    replacements = {
                        symbol: Mock(return_value=model)
                        for symbol in (
                            "Reranker",
                            "LiT5DistillReranker",
                            "LiT5ScoreReranker",
                            "SafeOpenai",
                            "SafeGenai",
                            "RankListwiseOSLLM",
                            "PointwiseVLLM",
                            "create_reranker",
                            "get_openai_api_key",
                            "get_genai_api_key",
                            "DataWriter",
                            "EvalFunction",
                            "ResponseAnalyzer",
                        )
                        if hasattr(demo, symbol)
                    }
                    argv = [name, *args] + (
                        [] if depth is None else ["--k", str(depth)]
                    )
                    with (
                        patch.multiple(demo, **replacements),
                        patch("sys.argv", argv),
                        patch("rank_llm.data.DataWriter"),
                        patch.object(Path, "mkdir"),
                        patch(
                            "rank_llm.retrieve.retriever.Retriever.from_dataset_with_prebuilt_index",
                            return_value=[request],
                        ) as retrieve,
                        redirect_stdout(io.StringIO()),
                    ):
                        demo.main()
                    self.assertEqual(retrieve.call_args.kwargs["k"], depth or 100)
                    for method, count in expected_calls.items():
                        calls = getattr(model, method).call_args_list
                        self.assertEqual(len(calls), count)
                        for call in calls:
                            self.assertEqual(call.kwargs["rank_end"], depth or 100)
                            self.assertEqual(
                                call.kwargs["top_k_retrieve"], depth or 100
                            )
                            self.assertTrue(call.kwargs["populate_invocations_history"])
                        if method != "rerank_batch":
                            self.assertEqual(getattr(model, method).await_count, count)

    def test_requested_depth_reaches_tail_sliding_windows(self):
        model = object.__new__(SafeOpenai)
        model._batch_size = 32
        model._window_size = 20
        model._stride = 10
        model.permutation_pipeline = Mock(
            side_effect=lambda result, *args, **kw: result
        )
        request = Request(
            query=Query(text="query", qid="1"),
            candidates=[Candidate(docid=str(i), score=1.0, doc={}) for i in range(200)],
        )
        model.rerank_batch([request], rank_end=200, top_k_retrieve=200)
        ranges = [call.args[1:3] for call in model.permutation_pipeline.call_args_list]
        self.assertEqual(ranges[0], (180, 200))
        self.assertEqual(
            set().union(*(set(range(*r)) for r in ranges)), set(range(200))
        )

    def test_short_requests_process_one_window_without_redundant_passes(self):
        for model_type in (SafeGenai, SafeOpenai):
            for size, start, depth, expected in (
                (0, 0, 15, []),
                (1, 0, 15, [(0, 1)]),
                (5, 0, 15, [(0, 5)]),
                (15, 0, 15, [(0, 15)]),
                (30, 0, 30, [(10, 30), (0, 20)]),
                (25, 0, 25, [(5, 25), (0, 15)]),
                (25, 10, 30, [(10, 25)]),
                (5, 10, 15, []),
            ):
                with self.subTest(model=model_type.__name__, size=size, start=start):
                    model = object.__new__(model_type)
                    model._batch_size, model._window_size, model._stride = 32, 20, 10
                    model.permutation_pipeline = Mock(
                        side_effect=lambda result, *a, **kw: result
                    )
                    request = Request(
                        query=Query(text="query", qid="1"),
                        candidates=[
                            Candidate(docid=str(i), score=1.0, doc={})
                            for i in range(size)
                        ],
                    )
                    model.rerank_batch(
                        [request],
                        rank_start=start,
                        rank_end=depth,
                        top_k_retrieve=depth,
                    )
                    self.assertEqual(
                        [
                            call.args[1:3]
                            for call in model.permutation_pipeline.call_args_list
                        ],
                        expected,
                    )

import copy
import importlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from rank_llm.data import Candidate, Query, Request, Result
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
            for depth, start, end, valid in (
                (None, 0, None, True),
                (10, 0, None, True),
                (200, 0, None, True),
                (200, 10, 50, True),
                (200, 10, None, True),
                (200, 0, 50, True),
                (200, -1, 50, False),
                (200, 50, 50, False),
                (200, 60, 50, False),
                (200, 0, 201, False),
                (200, 0, 0, False),
                (0, 0, None, False),
            ):
                with self.subTest(
                    demo=name, args=args, depth=depth, start=start, end=end
                ):
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
                    if start:
                        argv += ["--rank-start", str(start)]
                    if end is not None:
                        argv += ["--rank-end", str(end)]
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
                        if not valid:
                            with (
                                patch("sys.stderr", io.StringIO()),
                                self.assertRaises(SystemExit) as error,
                            ):
                                demo.main()
                            self.assertEqual(error.exception.code, 2)
                            retrieve.assert_not_called()
                            for replacement in replacements.values():
                                replacement.assert_not_called()
                            continue
                        demo.main()
                    self.assertEqual(retrieve.call_args.kwargs["k"], depth or 100)
                    for method, count in expected_calls.items():
                        calls = getattr(model, method).call_args_list
                        self.assertEqual(len(calls), count)
                        for call in calls:
                            self.assertEqual(call.kwargs["rank_start"], start)
                            self.assertEqual(
                                call.kwargs["rank_end"],
                                end if end is not None else depth or 100,
                            )
                            self.assertEqual(
                                call.kwargs["top_k_retrieve"], depth or 100
                            )
                            self.assertTrue(call.kwargs["populate_invocations_history"])
                        if method != "rerank_batch":
                            self.assertEqual(getattr(model, method).await_count, count)

    def test_requested_depth_reaches_tail_sliding_windows(self):
        request = Request(
            query=Query(text="query", qid="1"),
            candidates=[Candidate(docid=str(i), score=1.0, doc={}) for i in range(200)],
        )
        for window, stride, count in (
            (20, 10, 19),
            (12, 3, 64),
            (15, 9, 22),
            (20, 20, 10),
            (1, 1, 200),
        ):
            with self.subTest(window=window, stride=stride):
                model = object.__new__(SafeOpenai)
                model._batch_size, model._window_size, model._stride = (
                    32,
                    window,
                    stride,
                )
                model.permutation_pipeline = Mock(
                    side_effect=lambda result, *args, **kw: result
                )
                model.rerank_batch([request], rank_end=200, top_k_retrieve=200)
                ranges = [
                    call.args[1:3] for call in model.permutation_pipeline.call_args_list
                ]
                self.assertEqual(ranges[0], (200 - window, 200))
                self.assertEqual(ranges[-1][0], 0)
                self.assertEqual(len(ranges), count)
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

    def test_pointwise_demo_preserves_candidates_outside_selected_range(self):
        from rank_llm.demo import rerank_pointwise_vllm as demo

        def rerank(requests, **kwargs):
            selected = slice(kwargs["rank_start"], kwargs["rank_end"])
            candidates = copy.deepcopy(requests[0].candidates[selected][::-1])
            for candidate in candidates:
                candidate.score = float(candidate.docid)
            return [Result(query=requests[0].query, candidates=candidates)]

        for size, start, end in (
            (100, 10, 50),
            (20, 10, 50),
            (5, 10, 50),
            (100, 0, 100),
        ):
            request = Request(
                query=Query(text="query", qid="1"),
                candidates=[
                    Candidate(docid=str(i), score=float(size - i), doc={})
                    for i in range(size)
                ],
            )
            original = copy.deepcopy(request)
            with (
                self.subTest(size=size, start=start, end=end),
                tempfile.TemporaryDirectory() as output,
                patch.object(
                    demo.Retriever,
                    "from_dataset_with_prebuilt_index",
                    return_value=[request],
                ),
                patch.object(demo, "PointwiseVLLM"),
                patch.object(demo, "Reranker") as reranker,
                patch.object(demo, "_print_eval") as evaluate,
                patch.object(demo, "_print_sample") as sample,
                redirect_stdout(io.StringIO()),
            ):
                argv = f"demo --model test --k 100 --rank-start {start} --rank-end {end}".split()
                model = reranker.return_value
                model.rerank_batch.side_effect = rerank
                model.rerank_batch_async = AsyncMock(side_effect=rerank)
                with patch("sys.argv", [*argv, "--output-dir", output]):
                    demo.main()
                expected = list(range(size))
                expected[start:end] = expected[start:end][::-1]
                expected_ids = [str(i) for i in expected]
                expected_scores = (
                    [float(i) for i in expected]
                    if start == 0 and end >= size
                    else [c.score for c in original.candidates]
                )
                self.assertEqual(sample.call_count, 2)
                model.rerank_batch_async.assert_awaited_once()
                for call in sample.call_args_list + [evaluate.call_args]:
                    result = call.args[0][0]
                    self.assertEqual([c.docid for c in result.candidates], expected_ids)
                    self.assertEqual(
                        [c.score for c in result.candidates], expected_scores
                    )
                folder = Path(output) / "test" / "dl19"
                record = json.loads((folder / "rerank.jsonl").read_text())
                self.assertEqual(
                    [c["docid"] for c in record["candidates"]], expected_ids
                )
                rows = [
                    row.split()
                    for row in (folder / "rerank.txt").read_text().splitlines()
                ]
                self.assertEqual([row[2] for row in rows], expected_ids)
                ordered_rows = sorted(rows, key=lambda row: float(row[4]), reverse=True)
                self.assertEqual([row[2] for row in ordered_rows], expected_ids)
                self.assertEqual(request, original)

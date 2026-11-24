import asyncio
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM


def load_pipeline():
    # These tests exercise orchestration, without importing Java-backed retrieval.
    path = Path(__file__).resolve().parents[3] / "src/rank_llm/retrieve_and_rerank.py"
    spec = importlib.util.spec_from_file_location("_multipass_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"rank_llm.retrieve": MagicMock()}):
        spec.loader.exec_module(module)
    return module


class TestMultipassLifecycle(unittest.TestCase):
    def test_passes_share_loop_and_background_task(self):
        pipeline = load_pipeline()
        coordinator = MagicMock(spec=ListwiseRankLLM)
        loops, tasks, orders = [], [], []
        request = Request(
            Query("q", "1"),
            [
                Candidate("a", 1.0, {"text": "a"}),
                Candidate("b", 0.5, {"text": "b"}),
            ],
        )

        async def rerank(requests, *args, **kwargs):
            loops.append(asyncio.get_running_loop())
            orders.append([c.docid for c in requests[0].candidates])
            if not tasks:
                tasks.append(asyncio.create_task(asyncio.Event().wait()))
            await asyncio.sleep(0)
            self.assertFalse(tasks[0].done())
            return [Result(requests[0].query, list(reversed(requests[0].candidates)))]

        coordinator.rerank_batch_async = AsyncMock(side_effect=rerank)
        with (
            patch.object(
                pipeline.Reranker, "create_model_coordinator", return_value=coordinator
            ),
            patch.object(pipeline, "retrieve", return_value=[request]),
        ):
            pipeline.retrieve_and_rerank("model", "", [], num_passes=3)
        self.assertEqual(len(loops), 3)
        self.assertTrue(all(loop is loops[0] for loop in loops))
        self.assertEqual(orders, [["a", "b"], ["b", "a"], ["a", "b"]])
        self.assertTrue(tasks[0].cancelled())

    def test_failed_pass_does_not_write_or_evaluate(self):
        pipeline = load_pipeline()
        coordinator = MagicMock(spec=ListwiseRankLLM)
        coordinator.rerank_batch_async = AsyncMock(
            side_effect=RuntimeError("engine failed")
        )
        with (
            patch.object(
                pipeline.Reranker, "create_model_coordinator", return_value=coordinator
            ),
            patch.object(pipeline, "retrieve", return_value=[]),
            patch.object(pipeline.Reranker, "write_rerank_results") as write,
        ):
            with self.assertRaisesRegex(RuntimeError, "engine failed"):
                pipeline.retrieve_and_rerank("model", "", "dl19", num_passes=3)
            write.assert_not_called()
        self.assertEqual(coordinator.rerank_batch_async.await_count, 1)


class TestWindowFailure(unittest.IsolatedAsyncioTestCase):
    async def test_failure_cancels_other_windows(self):
        model = MagicMock(spec=ListwiseRankLLM)
        model._stride, model._window_size, model._batch_size = 1, 2, 2
        model._count_windows.return_value = 1
        model.create_prompt.return_value = ("prompt", 1)
        started, cancelled = asyncio.Event(), asyncio.Event()
        calls = 0

        async def generate(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                await started.wait()
                raise RuntimeError("engine failed")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        model.run_llm_async = generate
        request = Request(Query("q", "1"), [Candidate("a", 1.0, {"text": "a"})])
        with self.assertRaises(ExceptionGroup) as raised:
            await ListwiseRankLLM._sliding_windows_batched_async(
                model,
                [request, request],
                0,
                1,
                1,
                isolate_llm_slots=True,
            )
        self.assertIsInstance(raised.exception.exceptions[0], RuntimeError)
        self.assertTrue(cancelled.is_set())
        model._apply_llm_output_to_result.assert_not_called()

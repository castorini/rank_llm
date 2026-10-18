import importlib
import unittest
from unittest.mock import Mock, patch

from rank_llm.cli.operations import run_mcp_rerank
from rank_llm.rerank._model_lifecycle import close_owned_model

pipeline_module = importlib.import_module("rank_llm.retrieve_and_rerank")


class TestModelLifecycle(unittest.TestCase):
    def test_direct_rerank_closes_owned_model_on_success_and_failure(self):
        for failure in (None, ValueError("inference failed")):
            with self.subTest(failure=failure):
                coordinator = Mock()
                reranker = Mock()
                reranker.get_model_coordinator.return_value = coordinator
                reranker.rerank_batch.return_value = []
                reranker.rerank_batch.side_effect = failure
                with patch("rank_llm.cli.operations.Reranker", return_value=reranker):
                    if failure:
                        with self.assertRaises(ValueError) as raised:
                            run_mcp_rerank(
                                model_path="model", query_text="cats", candidates=[]
                            )
                        self.assertIs(raised.exception, failure)
                    else:
                        self.assertEqual(
                            run_mcp_rerank(
                                model_path="model", query_text="cats", candidates=[]
                            ),
                            [],
                        )
                coordinator.close.assert_called_once_with()

    def test_direct_rerank_closes_on_candidate_conversion_error(self):
        coordinator = Mock()
        reranker = Mock()
        reranker.get_model_coordinator.return_value = coordinator
        with patch("rank_llm.cli.operations.Reranker", return_value=reranker):
            with self.assertRaises(KeyError):
                run_mcp_rerank(model_path="model", query_text="cats", candidates=[{}])
        coordinator.close.assert_called_once_with()

    def test_direct_rerank_preserves_borrowed_model_even_on_failure(self):
        for failure in (None, ValueError("inference failed")):
            with self.subTest(failure=failure):
                reranker = Mock()
                reranker.rerank_batch.return_value = []
                reranker.rerank_batch.side_effect = failure
                try:
                    run_mcp_rerank(
                        model_path="model",
                        query_text="cats",
                        candidates=[],
                        reranker=reranker,
                    )
                except ValueError:
                    if failure is None:
                        raise
                reranker.get_model_coordinator.return_value.close.assert_not_called()

    def test_retrieval_ownership_and_interactive_transfer(self):
        for interactive, borrowed, failure in (
            (False, False, False),
            (False, False, True),
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (True, True, True),
            (False, True, False),
        ):
            with self.subTest(
                interactive=interactive, borrowed=borrowed, failure=failure
            ):
                default = Mock()
                # A noninteractive factory creates a new model even if a default was supplied.
                coordinator = default if borrowed and interactive else Mock()
                reranker = Mock()
                reranker.get_model_coordinator.return_value = coordinator
                reranker.rerank_batch.return_value = []
                error = ValueError("retrieval failed")
                with (
                    patch.object(pipeline_module, "Reranker", return_value=reranker),
                    patch.object(
                        pipeline_module,
                        "retrieve",
                        return_value=[],
                        side_effect=error if failure else None,
                    ),
                ):
                    options = dict(
                        model_path="model",
                        query="cats",
                        dataset=[],
                        interactive=interactive,
                        default_model_coordinator=default if borrowed else None,
                    )
                    if failure:
                        with self.assertRaises(ValueError) as raised:
                            pipeline_module.retrieve_and_rerank(**options)
                        self.assertIs(raised.exception, error)
                    else:
                        result = pipeline_module.retrieve_and_rerank(**options)
                        self.assertEqual(
                            result, ([], coordinator) if interactive else []
                        )
                should_close = coordinator is not (default if borrowed else None) and (
                    failure or not interactive
                )
                self.assertEqual(coordinator.close.call_count, int(should_close))
                if coordinator is not default:
                    default.close.assert_not_called()

    def test_retrieval_closes_owned_model_after_inference_or_output_failure(self):
        for stage in ("rerank_batch", "write_rerank_results"):
            with self.subTest(stage=stage):
                coordinator = Mock()
                reranker = Mock()
                reranker.get_model_coordinator.return_value = coordinator
                reranker.rerank_batch.return_value = []
                error = ValueError(f"{stage} failed")
                getattr(reranker, stage).side_effect = error
                with (
                    patch.object(pipeline_module, "Reranker", return_value=reranker),
                    patch.object(pipeline_module, "retrieve", return_value=[]),
                ):
                    with self.assertRaises(ValueError) as raised:
                        pipeline_module.retrieve_and_rerank(
                            model_path="model", query="cats", dataset="dataset"
                        )
                self.assertIs(raised.exception, error)
                coordinator.close.assert_called_once_with()

    def test_cleanup_error_does_not_replace_pipeline_error(self):
        coordinator = Mock()
        coordinator.close.side_effect = RuntimeError("cleanup failed")
        primary = ValueError("pipeline failed")
        with self.assertLogs("rank_llm.rerank._model_lifecycle", level="ERROR"):
            with self.assertRaises(ValueError) as raised:
                try:
                    raise primary
                finally:
                    close_owned_model(coordinator)
        self.assertIs(raised.exception, primary)

    def test_cleanup_error_propagates_after_success(self):
        coordinator = Mock()
        coordinator.close.side_effect = RuntimeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            close_owned_model(coordinator)

    def test_coordinator_without_close_is_supported(self):
        close_owned_model(None)
        close_owned_model(object())

import unittest
from unittest.mock import Mock

from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank.listwise.rank_gemini import SafeGenai
from rank_llm.rerank.listwise.rank_gpt import SafeOpenai


class RankingCostTest(unittest.TestCase):
    def model(self, model_type, window, stride):
        model = object.__new__(model_type)
        model._batch_size = 1
        model._window_size, model._stride, model._context_size = window, stride, 128
        model.permutation_pipeline = Mock(side_effect=lambda result, *a, **kw: result)
        model.create_prompt = Mock(return_value=("prompt", 20))
        model.get_num_tokens = Mock(return_value=20)
        model.num_output_tokens = Mock(return_value=10)
        model.cost_per_1k_token = Mock(
            side_effect=lambda input_token: 1.0 if input_token else 2.0
        )
        return model

    def test_prompt_estimates_match_inference_windows(self):
        for model_type in (SafeOpenai, SafeGenai):
            for size, start, end, window, stride in (
                (100, 0, 10, 20, 10),
                (5, 0, 15, 20, 10),
                (25, 0, 25, 20, 10),
                (30, 5, 30, 12, 3),
                (30, 5, 30, 15, 9),
                (40, 0, 40, 20, 20),
                (0, 0, 10, 20, 10),
                (5, 10, 15, 20, 10),
                (100, 10, 10, 20, 10),
            ):
                with self.subTest(
                    model=model_type.__name__,
                    size=size,
                    start=start,
                    end=end,
                    window=window,
                    stride=stride,
                ):
                    model = self.model(model_type, window, stride)
                    request = Request(
                        query=Query(text="query", qid="1"),
                        candidates=[
                            Candidate(docid=str(i), score=1.0, doc={})
                            for i in range(size)
                        ],
                    )
                    model.rerank_batch(
                        [request], rank_start=start, rank_end=end, top_k_retrieve=100
                    )
                    windows = [
                        call.args[1:3]
                        for call in model.permutation_pipeline.call_args_list
                    ]
                    cost, tokens = model.get_ranking_cost([request], start, end)
                    self.assertEqual(
                        [call.args[1:3] for call in model.create_prompt.call_args_list],
                        windows,
                    )
                    self.assertEqual(tokens, len(windows) * 30)
                    self.assertAlmostEqual(cost, len(windows) * 0.04)
                    self.assertEqual(model.get_ranking_cost([], start, end), (0.0, 0))

    def test_upper_bound_counts_complete_and_partial_windows(self):
        for model_type in (SafeOpenai, SafeGenai):
            for start, end, window, stride, count in (
                (0, 10, 20, 10, 1),
                (0, 5, 20, 10, 1),
                (0, 25, 20, 10, 2),
                (5, 30, 12, 3, 6),
                (5, 30, 15, 9, 3),
                (0, 40, 20, 20, 2),
                (10, 10, 20, 10, 0),
                (20, 10, 20, 10, 0),
                (0, 0, 20, 10, 0),
            ):
                with self.subTest(
                    model=model_type.__name__,
                    start=start,
                    end=end,
                    window=window,
                    stride=stride,
                ):
                    model = self.model(model_type, window, stride)
                    cost, tokens = model.get_ranking_cost_upperbound(2, start, end)
                    self.assertEqual(tokens, 2 * count * 128)
                    self.assertIsInstance(tokens, int)
                    self.assertAlmostEqual(cost, 2 * count * 0.138)
                    self.assertEqual(
                        model.get_ranking_cost_upperbound(0, start, end), (0.0, 0)
                    )

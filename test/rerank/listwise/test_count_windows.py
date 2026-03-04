import unittest

from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM


class TestCountWindows(unittest.TestCase):
    def test_incomplete_final_window_clamped_to_rank_start(self):
        """
        num_candidates=25 is not aligned with window_size=20 and stride=10.
        The last window slides past rank_start=0, so start_pos is clamped,
        producing an incomplete window.

        Window 1: [5, 25)   — full window of 20
        Slide:    end=15, raw start=-5 → clamped to 0
        Window 2: [0, 15)   — incomplete window of 15
        Total = 2.
        """
        result = ListwiseRankLLM._count_windows(
            num_candidates=25,
            rank_start=0,
            rank_end=100,
            window_size=20,
            stride=10,
        )
        self.assertEqual(result, 2)

    def test_incomplete_window_with_nonzero_rank_start(self):
        """
        Same idea but rank_start != 0.  The clamp prevents start_pos from
        going below rank_start.

        Window 1: [10, 30)  — full window of 20
        Slide:    end=20, raw start=0 → clamped to 5
        Window 2: [5, 20)   — incomplete window of 15
        Total = 2.
        """
        result = ListwiseRankLLM._count_windows(
            num_candidates=30,
            rank_start=5,
            rank_end=100,
            window_size=20,
            stride=10,
        )
        self.assertEqual(result, 2)

    def test_single_incomplete_window(self):
        """
        num_candidates=8 < window_size=20, so the very first window is
        already incomplete.  After one stride end goes negative and the
        loop exits.
        Total = 1.
        """
        result = ListwiseRankLLM._count_windows(
            num_candidates=8,
            rank_start=0,
            rank_end=100,
            window_size=20,
            stride=10,
        )
        self.assertEqual(result, 1)

    def test_exact_fit_no_incomplete_window(self):
        """
        Everything divides evenly — no incomplete window.

        Window 1: [20, 40)
        Window 2: [10, 30)
        Window 3: [0, 20)  — prev_start_pos == rank_start, loop exits.
        Total = 3.
        """
        result = ListwiseRankLLM._count_windows(
            num_candidates=40,
            rank_start=0,
            rank_end=100,
            window_size=20,
            stride=10,
        )
        self.assertEqual(result, 3)


if __name__ == "__main__":
    unittest.main()

import unittest

from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM


class DummyListwiseRankLLM(ListwiseRankLLM):
    def __init__(self, context_size=4096, output_tokens=40):
        self._context_size = context_size
        self._output_tokens = output_tokens

    def run_llm_batched(self, prompts, **kwargs):
        raise NotImplementedError

    def run_llm(self, prompt, **kwargs):
        raise NotImplementedError

    def rerank_batch(self, requests, **kwargs):
        raise NotImplementedError

    def create_prompt_batched(self, results, rank_start, rank_end):
        raise NotImplementedError

    def create_prompt(self, result, rank_start, rank_end):
        raise NotImplementedError

    def get_num_tokens(self, prompt):
        raise NotImplementedError

    def cost_per_1k_token(self, input_token):
        return 0

    def num_output_tokens(self, current_window_size=None):
        return self._output_tokens


class PassageTruncationTest(unittest.TestCase):
    def test_estimate_uses_context_output_and_window(self):
        reranker = DummyListwiseRankLLM(context_size=4096, output_tokens=40)

        self.assertEqual(reranker.estimate_max_passage_words(20), 152)
        self.assertEqual(reranker.estimate_max_passage_words(10), 304)

    def test_reduction_uses_token_to_word_ratio(self):
        reranker = DummyListwiseRankLLM(context_size=4096, output_tokens=40)

        self.assertEqual(
            reranker.reduce_max_passage_words(
                current_max_words=300,
                current_window_size=20,
                prompt_tokens=4500,
            ),
            283,
        )

    def test_estimate_rejects_empty_window(self):
        reranker = DummyListwiseRankLLM()

        with self.assertRaisesRegex(ValueError, "current_window_size"):
            reranker.estimate_max_passage_words(0)

    def test_reduction_rejects_unrepresentable_prompt(self):
        reranker = DummyListwiseRankLLM(context_size=128, output_tokens=40)

        with self.assertRaisesRegex(ValueError, "one word per passage"):
            reranker.reduce_max_passage_words(
                current_max_words=1,
                current_window_size=20,
                prompt_tokens=200,
            )


if __name__ == "__main__":
    unittest.main()

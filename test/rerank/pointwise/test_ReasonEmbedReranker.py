import asyncio
import unittest

from rank_llm.rerank.pointwise.reason_embed_reranker import ReasonEmbedReranker


class _FakeInferenceHandler:
    def _format_template(self, template_key, fmt_values):
        return f"Query={fmt_values['query']} Doc={fmt_values['doc_content']}"


class TestReasonEmbedReranker(unittest.TestCase):
    def test_parse_score_from_tags(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)
        self.assertEqual(reranker._parse_score("foo <score>87</score> bar"), 87.0)

    def test_parse_score_without_tags_is_zero(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)
        # A response with numbers but no score tags should not be parsed as a score.
        response = "1. Query Analysis\n2. Document Analysis\n3. Relevance Annotation"
        self.assertEqual(reranker._parse_score(response), 0.0)

    def test_build_user_input_appends_strict_output_instruction(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)
        reranker._inference_handler = _FakeInferenceHandler()
        prompt = reranker._build_user_input("q", "d")
        self.assertIn("Query=q Doc=d", prompt)
        self.assertIn("IMPORTANT OUTPUT FORMAT", prompt)
        self.assertIn("<score>INTEGER_0_TO_100</score>", prompt)

    def test_score_candidates_batched_returns_prompt_and_token_counts(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)

        reranker._prepare_prompt = lambda query, doc: f"prompt::{query}::{doc}"

        async def _fake_run_prompts_async(prompts):
            return [("<score>42</score>", 10, 3) for _ in prompts]

        reranker._run_prompts_async = _fake_run_prompts_async
        scored = asyncio.run(
            reranker._score_candidates_batched_async("q", ["d1", "d2"])
        )

        self.assertEqual(len(scored), 2)
        self.assertEqual(scored[0][0], "prompt::q::d1")
        self.assertEqual(scored[0][1], 10)
        self.assertEqual(scored[0][2], 3)
        self.assertEqual(scored[0][3], 42.0)


if __name__ == "__main__":
    unittest.main()

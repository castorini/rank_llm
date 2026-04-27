import asyncio
import unittest

from rank_llm.data import Candidate, Query, Result
from rank_llm.rerank.pointwise.reason_embed_reranker import ReasonEmbedReranker


class _FakeInferenceHandler:
    def format_body(self, query, doc_content, **extra_values):
        return (
            f"Query={query} Doc={doc_content} "
            f"Rel={extra_values['relevance_definition']} "
            f"QueryType={extra_values['query_type']} "
            f"DocType={extra_values['doc_type']}"
        )


class _FakeTokenizer:
    chat_template = "fake-template"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"tok{i}" for i in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        rendered_messages = " || ".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        return (
            f"CHAT::{rendered_messages}::gen={add_generation_prompt}"
            f"::think={enable_thinking}"
        )


class TestReasonEmbedReranker(unittest.TestCase):
    def test_parse_score_from_tags(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)
        self.assertEqual(reranker._parse_score("foo <score>87</score> bar"), 87.0)

    def test_parse_score_uses_last_tag(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)
        response = "<score>12</score>\nreasoning\n<score>65</score>"
        self.assertEqual(reranker._parse_score(response), 65.0)

    def test_parse_score_without_tags_is_zero(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)
        response = "1. Query Analysis\n2. Document Analysis\n3. Relevance Annotation"
        self.assertEqual(reranker._parse_score(response), 0.0)

    def test_build_user_input_uses_template_verbatim(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)
        reranker._inference_handler = _FakeInferenceHandler()
        reranker._relevance_definition = "rel"
        reranker._query_type = "question"
        reranker._doc_type = "passage"
        prompt = reranker._build_user_input("q", "d")
        self.assertEqual(
            prompt,
            "Query=q Doc=d Rel=rel QueryType=question DocType=passage",
        )

    def test_prepare_prompt_uses_single_user_turn_and_disables_thinking(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)
        reranker._inference_handler = _FakeInferenceHandler()
        reranker._tokenizer = _FakeTokenizer()
        reranker._relevance_definition = "rel"
        reranker._query_type = "question"
        reranker._doc_type = "passage"
        reranker._context_size = 128
        reranker._max_new_tokens = 16
        prompt = reranker._prepare_prompt("q", "d")
        self.assertIn(
            "CHAT::user:Query=q Doc=d Rel=rel QueryType=question DocType=passage",
            prompt,
        )
        self.assertIn("think=False", prompt)
        self.assertNotIn("system:", prompt)

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

    def test_sort_reranked_slice_preserves_outside_range(self):
        reranker = ReasonEmbedReranker.__new__(ReasonEmbedReranker)
        result = Result(
            query=Query(text="q", qid="1"),
            candidates=[
                Candidate(docid="a", score=100.0, doc={"text": "a"}),
                Candidate(docid="b", score=30.0, doc={"text": "b"}),
                Candidate(docid="c", score=90.0, doc={"text": "c"}),
                Candidate(docid="d", score=80.0, doc={"text": "d"}),
            ],
            invocations_history=[],
        )

        reranker._sort_reranked_slice(result, rank_start=1, rank_end=3)

        self.assertEqual(
            [candidate.docid for candidate in result.candidates],
            ["a", "c", "b", "d"],
        )


if __name__ == "__main__":
    unittest.main()

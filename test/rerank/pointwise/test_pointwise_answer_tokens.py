"""Positive / negative answer tokens for pointwise vLLM scoring."""

import asyncio
import math
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from dacite import from_dict

from rank_llm.data import Request
from rank_llm.rerank.pointwise.pointwise_inference_handler import (
    PointwiseInferenceHandler,
    validate_output_tokens,
)
from rank_llm.rerank.pointwise.pointwise_vllm import (
    DEFAULT_NEGATIVE_TOKENS,
    DEFAULT_POSITIVE_TOKENS,
    PointwiseVLLM,
)
from rank_llm.rerank.vllm_handler_with_openai_sdk import (
    PointwiseAnswerTokens,
    VllmHandlerWithOpenAISDK,
)

# Single tokens; any other text encodes to one id per character (multi-token).
VOCAB = {
    "yes": 1,
    "Yes": 2,
    "YES": 3,
    " yes": 4,
    " Yes": 5,
    " YES": 6,
    "no": 11,
    "No": 12,
    "NO": 13,
    " no": 14,
    " No": 15,
    " NO": 16,
    "true": 21,
    "True": 22,
    "false": 31,
}
INVERSE = {i: s for s, i in VOCAB.items()}


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        if text in VOCAB:
            return [VOCAB[text]]
        max_length = kwargs.get("max_length")
        pieces = [1000 + ord(c) for c in text] if max_length is None else text.split()
        return pieces if max_length is None else pieces[: int(max_length)]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(INVERSE.get(i, " " + str(i)) for i in ids)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=True, **kw
    ):
        return [0] * 16


def lp(token, prob):
    return {"token": token, "logprob": math.log(prob)}


class TestPointwiseAnswerTokens(unittest.TestCase):
    def test_true_false_answer_tokens(self):
        tokens = PointwiseAnswerTokens.from_spellings(
            FakeTokenizer(), ["true", "True"], ["false"]
        )
        self.assertEqual(tokens.positive_ids, (21, 22))
        self.assertEqual(tokens.negative_ids, (31,))
        self.assertEqual(tokens.allowed_token_ids, [21, 22, 31])

    def test_multi_token_spellings_are_skipped_with_a_warning(self):
        with self.assertLogs(
            "rank_llm.rerank.vllm_handler_with_openai_sdk", level="WARNING"
        ) as logs:
            tokens = PointwiseAnswerTokens.from_spellings(
                FakeTokenizer(), ["true", "relevant"], ["false"]
            )
        self.assertEqual(tokens.positive_ids, (21,))
        self.assertIn("relevant", logs.output[0])

    def test_a_side_without_single_tokens_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "No negative answer spelling"):
            PointwiseAnswerTokens.from_spellings(
                FakeTokenizer(), ["true"], ["not relevant"]
            )

    def test_a_token_on_both_sides_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            PointwiseAnswerTokens.from_spellings(
                FakeTokenizer(), ["true", "yes"], ["false", "yes"]
            )

    def test_score_by_token_id(self):
        tokens = PointwiseAnswerTokens.from_spellings(
            FakeTokenizer(), ["true", "True"], ["false"]
        )
        score, pos_lp, neg_lp = tokens.score(
            [
                lp("token_id:21", 0.5),
                lp("token_id:22", 0.1),
                lp("token_id:31", 0.2),
                lp("token_id:999", 0.2),  # not an answer token: ignored
            ]
        )
        self.assertAlmostEqual(score, 0.6 / 0.8)
        self.assertAlmostEqual(pos_lp, math.log(0.6))
        self.assertAlmostEqual(neg_lp, math.log(0.2))

    def test_score_by_decoded_text_when_the_server_returns_text(self):
        tokens = PointwiseAnswerTokens.from_spellings(
            FakeTokenizer(), ["true"], ["false"]
        )
        score, _, _ = tokens.score([lp("true", 0.3), lp("false", 0.1)])
        self.assertAlmostEqual(score, 0.75)
        # Exact text only: no stripping or case folding.
        self.assertEqual(tokens.score([lp(" true", 0.3), lp("False", 0.1)])[0], 0.0)

    def test_no_answer_mass_scores_zero(self):
        tokens = PointwiseAnswerTokens.from_spellings(
            FakeTokenizer(), ["true"], ["false"]
        )
        self.assertEqual(tokens.score([]), (0.0, -20.0, -20.0))

    def test_default_lists_match_the_legacy_yes_no_scoring(self):
        tokens = PointwiseAnswerTokens.from_spellings(
            FakeTokenizer(), DEFAULT_POSITIVE_TOKENS, DEFAULT_NEGATIVE_TOKENS
        )
        # Bare and leading-space forms are single tokens here; trailing-space
        # forms are skipped, as with real BPE tokenizers.
        self.assertEqual(tokens.positive_ids, (1, 2, 3, 4, 5, 6))
        self.assertEqual(tokens.negative_ids, (11, 12, 13, 14, 15, 16))
        cases = [
            [lp("yes", 0.7), lp("No", 0.2), lp(" no", 0.05)],
            [lp(" Yes", 0.01), lp("NO", 0.9)],
            [lp("YES", 0.4), lp(" yes", 0.3), lp("no", 0.2), lp(" NO", 0.1)],
        ]
        for top in cases:
            legacy = VllmHandlerWithOpenAISDK.score_from_top_logprobs(top)
            by_text = tokens.score(top)
            by_id = tokens.score(
                [
                    lp(f"token_id:{VOCAB[e['token']]}", math.exp(e["logprob"]))
                    for e in top
                ]
            )
            for new in (by_text, by_id):
                for a, b in zip(legacy, new, strict=False):
                    self.assertAlmostEqual(a, b)


class TestOutputTokensTemplateKey(unittest.TestCase):
    TEMPLATE = {
        "method": "pointwise",
        "message_roles": "reranker",
        "instruction": "Judge relevance; answer true or false.",
        "output_tokens": {"positive": ["true"], "negative": ["false"]},
    }

    def test_a_template_may_declare_its_answer_tokens(self):
        handler = PointwiseInferenceHandler(dict(self.TEMPLATE))
        self.assertEqual(handler.template["output_tokens"]["positive"], ["true"])

    def test_malformed_output_tokens_are_rejected(self):
        for bad in (
            {"positive": ["true"]},
            {"positive": ["true"], "negative": []},
            {"positive": ["true"], "negative": [""]},
            {"positive": "true", "negative": ["false"]},
            {"positive": ["true"], "negative": ["false"], "neutral": ["maybe"]},
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_output_tokens(bad)
            with self.assertRaises(ValueError, msg=repr(bad)):
                PointwiseInferenceHandler({**self.TEMPLATE, "output_tokens": bad})


TRUE_FALSE_TEMPLATE = """method: "pointwise"
message_roles: "reranker"
instruction: "Judge relevance; answer true or false."
output_tokens:
  positive: ["true", "True"]
  negative: ["false"]
"""


class TestPointwiseVLLMAnswerTokens(unittest.TestCase):
    def setUp(self):
        self.mock_vllm = MagicMock()
        self.mock_vllm.get_tokenizer.return_value = FakeTokenizer()
        self.mock_vllm.chat_completion_score_async = AsyncMock(
            return_value=("true", 1, 0.8, {"prompt_tokens": 4, "completion_tokens": 1})
        )
        handle, self.template_path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(handle, "w") as f:
            f.write(TRUE_FALSE_TEMPLATE)

    def tearDown(self):
        os.unlink(self.template_path)

    def make(self, **kwargs) -> PointwiseVLLM:
        with patch(
            "rank_llm.rerank.pointwise.pointwise_vllm.VllmHandlerWithOpenAISDK",
            return_value=self.mock_vllm,
        ):
            return PointwiseVLLM(
                model="dummy-model", base_url="http://127.0.0.1:8000/v1", **kwargs
            )

    def test_defaults_are_the_yes_no_spellings(self):
        reranker = self.make()
        self.assertEqual(reranker.answer_tokens.positive_ids, (1, 2, 3, 4, 5, 6))
        self.assertEqual(
            reranker._score_extra_body["allowed_token_ids"],
            [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],
        )

    def test_template_output_tokens_are_used(self):
        reranker = self.make(prompt_template_path=self.template_path)
        self.assertEqual(reranker.answer_tokens.positive_ids, (21, 22))
        self.assertEqual(reranker.answer_tokens.negative_ids, (31,))
        self.assertEqual(reranker._score_extra_body["allowed_token_ids"], [21, 22, 31])

    def test_arguments_override_the_template(self):
        reranker = self.make(
            prompt_template_path=self.template_path,
            positive_tokens=["yes"],
            negative_tokens=["no"],
        )
        self.assertEqual(reranker.answer_tokens.allowed_token_ids, [1, 11])

    def test_both_sides_must_be_given_together(self):
        with self.assertRaisesRegex(ValueError, "together"):
            self.make(positive_tokens=["yes"])

    def test_score_calls_carry_the_answer_tokens(self):
        reranker = self.make(prompt_template_path=self.template_path)
        request = from_dict(
            data_class=Request,
            data={
                "query": {"text": "q", "qid": "1"},
                "candidates": [{"doc": {"contents": "d"}, "docid": "d1", "score": 1.0}],
            },
        )
        reranker.rerank_batch([request])
        kwargs = self.mock_vllm.chat_completion_score_async.call_args.kwargs
        self.assertIs(kwargs["answer_tokens"], reranker.answer_tokens)
        self.assertEqual(kwargs["top_logprobs"], 20)
        self.assertEqual(kwargs["extra_body"]["allowed_token_ids"], [21, 22, 31])


class TestHandlerRequestsTokenIds(unittest.TestCase):
    def test_answer_tokens_request_and_score_token_ids(self):
        with (
            patch("rank_llm.rerank.vllm_handler_with_openai_sdk.OpenAI"),
            patch(
                "rank_llm.rerank.vllm_handler_with_openai_sdk.AsyncOpenAI"
            ) as async_cls,
            patch(
                "rank_llm.rerank.vllm_handler_with_openai_sdk.AutoTokenizer"
            ) as tok_cls,
        ):
            tok_cls.from_pretrained.return_value = FakeTokenizer()
            handler = VllmHandlerWithOpenAISDK(base_url="http://x/v1", model="m")
            entries = []
            for token, prob in (("token_id:21", 0.6), ("token_id:31", 0.2)):
                e = MagicMock()
                e.token, e.logprob = token, math.log(prob)
                entries.append(e)
            choice = MagicMock()
            choice.message.content = "true"
            choice.logprobs.content = [MagicMock(top_logprobs=entries)]
            response = MagicMock(choices=[choice])
            response.usage.model_dump.return_value = {"completion_tokens": 1}
            create = AsyncMock(return_value=response)
            async_cls.return_value.chat.completions.create = create
            handler._async_client = async_cls.return_value
            tokens = PointwiseAnswerTokens.from_spellings(
                FakeTokenizer(), ["true"], ["false"]
            )

            _, _, score, _ = asyncio.run(
                handler.chat_completion_score_async(
                    [{"role": "user", "content": "hi"}],
                    extra_body={"allowed_token_ids": tokens.allowed_token_ids},
                    answer_tokens=tokens,
                )
            )
        self.assertAlmostEqual(score, 0.75)
        sent = create.call_args.kwargs["extra_body"]
        self.assertTrue(sent["return_tokens_as_token_ids"])
        self.assertEqual(sent["allowed_token_ids"], [21, 31])


if __name__ == "__main__":
    unittest.main()

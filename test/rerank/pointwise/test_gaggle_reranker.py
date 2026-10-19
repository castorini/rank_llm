import unittest
from unittest.mock import MagicMock, patch

import torch

from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank.pointwise.gaggle_reranker import GaggleReranker
from rank_llm.rerank.reranker import Reranker


class TestGaggleReranker(unittest.TestCase):
    def test_rejects_custom_prompt_template(self):
        with self.assertRaisesRegex(ValueError, "fixed prompt template"):
            GaggleReranker(prompt_template_path="custom.yaml")

    def test_accepts_relative_bundled_prompt_template(self):
        with (
            patch(
                "rank_llm.rerank.pointwise.gaggle_reranker.AutoModelForSequenceClassification"
            ),
            patch("rank_llm.rerank.pointwise.gaggle_reranker.AutoTokenizer"),
        ):
            GaggleReranker(
                prompt_template_path="src/rank_llm/rerank/prompt_templates/gaggle_template.yaml",
                device="cpu",
            )

    @patch("rank_llm.rerank.pointwise.gaggle_reranker.torch.cuda.is_bf16_supported")
    @patch(
        "rank_llm.rerank.pointwise.gaggle_reranker.AutoModelForSequenceClassification"
    )
    @patch("rank_llm.rerank.pointwise.gaggle_reranker.AutoTokenizer")
    def test_uses_fp16_without_bf16_support(
        self, tokenizer_cls, model_cls, bf16_supported
    ):
        bf16_supported.return_value = False
        GaggleReranker(device="cuda")
        self.assertEqual(
            model_cls.from_pretrained.call_args.kwargs["torch_dtype"], torch.float16
        )

    @patch(
        "rank_llm.rerank.pointwise.gaggle_reranker.AutoModelForSequenceClassification"
    )
    @patch("rank_llm.rerank.pointwise.gaggle_reranker.AutoTokenizer")
    def test_factory_scores_and_sorts_pairs(self, tokenizer_cls, model_cls):
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        model.build_input_ids.side_effect = lambda tokenizer, query, passage: list(
            range(len(query) + len(passage))
        )
        model.compute_score.return_value = [2.0, -1.0, 0.5]
        model_cls.from_pretrained.return_value = model

        coordinator = Reranker.create_model_coordinator(
            "castorini/gaggle-reranker-20261005",
            None,
            False,
            device="cpu",
            batch_size=3,
        )
        self.assertIsInstance(coordinator, GaggleReranker)
        request = Request(
            Query("query with Passage: marker", "q1"),
            [
                Candidate("a", 0, {"text": "first Relevant: text"}),
                Candidate("b", 0, {"contents": "second"}),
                Candidate("c", 0, {"title": "Third", "body": "last"}),
            ],
        )
        result = coordinator.rerank_batch([request], populate_invocations_history=True)[
            0
        ]

        self.assertEqual(
            [candidate.docid for candidate in result.candidates], ["a", "c", "b"]
        )
        model.compute_score.assert_called_once_with(
            [
                ("query with Passage: marker", "first Relevant: text"),
                ("query with Passage: marker", "second"),
                ("query with Passage: marker", "Title: Third Content: last"),
            ],
            tokenizer=tokenizer_cls.from_pretrained.return_value,
            batch_size=3,
        )
        self.assertEqual(len(result.invocations_history), 3)


if __name__ == "__main__":
    unittest.main()

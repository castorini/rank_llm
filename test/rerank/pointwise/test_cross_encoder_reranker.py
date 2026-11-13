import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank.pointwise.cross_encoder_reranker import CrossEncoderReranker
from rank_llm.rerank.reranker import Reranker


@patch(
    "rank_llm.rerank.pointwise.cross_encoder_reranker.AutoModelForSequenceClassification"
)
@patch("rank_llm.rerank.pointwise.cross_encoder_reranker.AutoTokenizer")
class TestCrossEncoderReranker(unittest.TestCase):
    def test_factory_creation(self, mock_tok, mock_model):
        coordinator = Reranker.create_model_coordinator(
            "cross-encoder", None, False, device="cpu"
        )
        self.assertIsInstance(coordinator, CrossEncoderReranker)
        self.assertEqual(coordinator._model, "cross-encoder/ms-marco-MiniLM-L-6-v2")

    def test_bge_reranker_factory(self, mock_tok, mock_model):
        coordinator = Reranker.create_model_coordinator(
            "bge-reranker", None, False, device="cpu"
        )
        self.assertIsInstance(coordinator, CrossEncoderReranker)
        self.assertEqual(coordinator._model, "BAAI/bge-reranker-base")

    def test_prompt_creation_and_parsing(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        mock_tokenizer_inst.encode.return_value = [101, 102]
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.return_value = SimpleNamespace(
            logits=torch.tensor([[2.0], [1.0]])
        )
        mock_model.from_pretrained.return_value = mock_model_inst

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        query = Query(text="What is hybrid search?", qid="q1")
        candidates = [
            Candidate(
                doc={"text": "Hybrid search combines dense and sparse IR."},
                docid="d1",
                score=0.0,
            ),
            Candidate(
                doc={"contents": "Reciprocal rank fusion merges rank positions."},
                docid="d2",
                score=0.0,
            ),
        ]
        request = Request(query=query, candidates=candidates)

        results = reranker.rerank_batch([request])
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].candidates), 2)

    def test_document_extraction_with_segment_and_title(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.encode.return_value = [101, 102, 103]
        mock_tokenizer_inst.decode.return_value = (
            "Title: Sample Paper Content: Segment details"
        )
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        query = Query(text="Sample Query", qid="q1")
        candidate = Candidate(
            doc={"segment": "Segment details", "title": "Sample Paper"},
            docid="d1",
            score=0.0,
        )
        from rank_llm.data import Result

        res = Result(query=query, candidates=[candidate])

        prompt, num_tokens = reranker.create_prompt(res, 0)
        self.assertIn("Title: Sample Paper", prompt)
        self.assertIn("Segment details", prompt)

    def test_binary_logits_relative_ranking(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        mock_tokenizer_inst.encode.return_value = [1, 2]
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        # Candidate 0: [100.0, 1.0] -> overwhelming class 0, class 1 probability ~ 0
        # Candidate 1: [0.0, 0.0] -> neutral, class 1 probability = 0.5
        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.return_value = SimpleNamespace(
            logits=torch.tensor([[100.0, 1.0], [0.0, 0.0]])
        )
        mock_model.from_pretrained.return_value = mock_model_inst

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        prompts = [
            "QUERY: q DOCUMENT: d1",
            "QUERY: q DOCUMENT: d2",
        ]
        outputs, tokens, scores = reranker.run_llm_batched(prompts)

        self.assertEqual(len(scores), 2)
        # Candidate 1 must score higher than candidate 0 because candidate 0 has class 0 overwhelming
        self.assertGreater(scores[1], scores[0])
        self.assertAlmostEqual(scores[1], 0.5, places=3)
        self.assertLess(scores[0], 0.01)

    def test_context_size_clamping(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.model_max_length = 512
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        mock_model_inst = MagicMock()
        mock_model_inst.config = SimpleNamespace(max_position_embeddings=512)
        mock_model_inst.to.return_value = mock_model_inst
        mock_model.from_pretrained.return_value = mock_model_inst

        # Request context_size=4096, should be clamped to 512
        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            context_size=4096,
            device="cpu",
        )
        self.assertEqual(reranker._context_size, 512)

    def test_custom_template_rejection(self, mock_tok, mock_model):
        mock_tok.from_pretrained.return_value = MagicMock()
        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model.from_pretrained.return_value = mock_model_inst

        with self.assertRaises(ValueError):
            CrossEncoderReranker(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                prompt_template_path="some_custom_template.yaml",
                device="cpu",
            )

    def test_single_logit_shape(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer_inst.encode.return_value = [1, 2]
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.return_value = SimpleNamespace(logits=torch.tensor([[3.2]]))
        mock_model.from_pretrained.return_value = mock_model_inst

        reranker = CrossEncoderReranker(
            model="BAAI/bge-reranker-base", device="cpu", apply_sigmoid=False
        )

        prompts = ["QUERY: test query DOCUMENT: test passage 1"]
        outputs, tokens, scores = reranker.run_llm_batched(prompts)
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 3.2, places=4)


if __name__ == "__main__":
    unittest.main()

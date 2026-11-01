import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank.pointwise.cross_encoder_reranker import CrossEncoderReranker
from rank_llm.rerank.reranker import TEMPLATES, Reranker


@patch("rank_llm.rerank.pointwise.cross_encoder_reranker.AutoModelForSequenceClassification")
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
        mock_tokenizer_inst.encode.return_value = [101, 102]
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        query = Query(text="What is hybrid search?", qid="q1")
        candidates = [
            Candidate(doc={"text": "Hybrid search combines dense and sparse IR."}, docid="d1", score=0.0),
            Candidate(doc={"contents": "Reciprocal rank fusion merges rank positions."}, docid="d2", score=0.0),
        ]
        request = Request(query=query, candidates=candidates)

        results = reranker.rerank_batch([request])
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].candidates), 2)

    def test_batched_scoring_shapes(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        mock_tokenizer_inst.encode.return_value = [1, 2]
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        # Case 1: Binary logits [batch_size, 2]
        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.return_value = SimpleNamespace(
            logits=torch.tensor([[-1.5, 2.5], [1.0, -0.5]])
        )
        mock_model.from_pretrained.return_value = mock_model_inst

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", apply_sigmoid=True
        )

        prompts = [
            "QUERY: test query DOCUMENT: test passage 1",
            "QUERY: test query DOCUMENT: test passage 2",
        ]
        outputs, tokens, scores = reranker.run_llm_batched(prompts)

        self.assertEqual(len(scores), 2)
        # 2.5 passed through sigmoid should be > 0.9
        self.assertGreater(scores[0], 0.9)
        # -0.5 passed through sigmoid should be < 0.4
        self.assertLess(scores[1], 0.4)

    def test_single_logit_shape(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer_inst.encode.return_value = [1, 2]
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        # Case 2: Single logit [batch_size, 1]
        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.return_value = SimpleNamespace(
            logits=torch.tensor([[3.2]])
        )
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

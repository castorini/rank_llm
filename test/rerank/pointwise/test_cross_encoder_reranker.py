import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.rerank.pointwise.cross_encoder_reranker import CrossEncoderReranker
from rank_llm.rerank.reranker import Reranker


@patch(
    "rank_llm.rerank.pointwise.cross_encoder_reranker.AutoModelForSequenceClassification"
)
@patch("rank_llm.rerank.pointwise.cross_encoder_reranker.AutoTokenizer")
class TestCrossEncoderReranker(unittest.TestCase):
    def test_factory_creation(self, mock_tok, mock_model):
        coordinator = Reranker.create_model_coordinator(
            "cross-encoder", None, False, device="cpu", max_passage_words=500
        )
        self.assertIsInstance(coordinator, CrossEncoderReranker)
        self.assertEqual(coordinator._model, "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertEqual(coordinator._max_passage_words, 500)

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
        res = Result(query=query, candidates=[candidate])

        prompt, num_tokens = reranker.create_prompt(res, 0)
        q, d = reranker._parse_prompt(prompt)
        self.assertIn("Title: Sample Paper", d)
        self.assertIn("Segment details", d)

    def test_query_containing_sentinel_preserved(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.encode.return_value = [101, 102]
        mock_tokenizer_inst.decode.return_value = "Normal document body"
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        query_text = "What does DOCUMENT: mean in syntax?"
        query = Query(text=query_text, qid="q1")
        candidate = Candidate(
            doc={"text": "Normal document body"}, docid="d1", score=0.0
        )
        res = Result(query=query, candidates=[candidate])

        prompt, _ = reranker.create_prompt(res, 0)
        parsed_q, parsed_d = reranker._parse_prompt(prompt)
        self.assertEqual(parsed_q, query_text)
        self.assertEqual(parsed_d, "Normal document body")

    def test_binary_logits_relative_ranking(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        mock_tokenizer_inst.encode.return_value = [1, 2]
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

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
            '{"query": "q", "doc": "d1"}',
            '{"query": "q", "doc": "d2"}',
        ]
        outputs, tokens, scores = reranker.run_llm_batched(prompts)

        self.assertEqual(len(scores), 2)
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

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            context_size=4096,
            device="cpu",
        )
        self.assertEqual(reranker._context_size, 512)

    def test_bundled_template_path_accepted(self, mock_tok, mock_model):
        mock_tok.from_pretrained.return_value = MagicMock()
        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model.from_pretrained.return_value = mock_model_inst

        # Should accept relative path to bundled template without raising ValueError
        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            prompt_template_path="src/rank_llm/rerank/prompt_templates/cross_encoder_template.yaml",
            device="cpu",
        )
        self.assertIsNotNone(reranker)

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

        prompts = ['{"query": "test query", "doc": "test passage 1"}']
        outputs, tokens, scores = reranker.run_llm_batched(prompts)
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 3.2, places=4)

    def test_trust_remote_code_forwarded(self, mock_tok, mock_model):
        mock_tok.from_pretrained.return_value = MagicMock()
        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model.from_pretrained.return_value = mock_model_inst

        coordinator = Reranker.create_model_coordinator(
            "cross-encoder", None, False, device="cpu", trust_remote_code=True
        )
        self.assertIsInstance(coordinator, CrossEncoderReranker)
        mock_tok.from_pretrained.assert_called_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", trust_remote_code=True
        )
        mock_model.from_pretrained.assert_called_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", trust_remote_code=True
        )

    def test_multiclass_explicit_relevance_label_id(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        # Class logits: [contradiction=0.0, entailment=10.0, neutral=0.0]
        mock_model_inst.return_value = SimpleNamespace(
            logits=torch.tensor([[0.0, 10.0, 0.0]])
        )
        mock_model.from_pretrained.return_value = mock_model_inst

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            relevance_label_id=1,
        )

        prompts = ['{"query": "q", "doc": "d"}']
        outputs, tokens, scores = reranker.run_llm_batched(prompts)
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0.99)

    def test_multiclass_auto_discovered_label(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.config = SimpleNamespace(
            label2id={"contradiction": 0, "entailment": 1, "neutral": 2}
        )
        mock_model_inst.return_value = SimpleNamespace(
            logits=torch.tensor([[0.0, 10.0, 0.0]])
        )
        mock_model.from_pretrained.return_value = mock_model_inst

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
        )
        self.assertEqual(reranker._relevance_label_id, 1)

        prompts = ['{"query": "q", "doc": "d"}']
        outputs, tokens, scores = reranker.run_llm_batched(prompts)
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0.99)

    def test_multiclass_unresolved_label_raises_value_error(self, mock_tok, mock_model):
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer_inst.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tok.from_pretrained.return_value = mock_tokenizer_inst

        mock_model_inst = MagicMock()
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.config = SimpleNamespace(label2id={"a": 0, "b": 1, "c": 2})
        mock_model_inst.return_value = SimpleNamespace(
            logits=torch.tensor([[1.0, 2.0, 3.0]])
        )
        mock_model.from_pretrained.return_value = mock_model_inst

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
        )

        prompts = ['{"query": "q", "doc": "d"}']
        with self.assertRaises(ValueError) as ctx:
            reranker.run_llm_batched(prompts)
        self.assertIn("relevance label could be identified", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

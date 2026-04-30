import copy
import unittest
from unittest.mock import MagicMock, patch

from dacite import from_dict

from rank_llm.data import Request, Result


def _make_request(num_candidates=5, qid=1, query_text="what is deep learning"):
    candidates = []
    for i in range(num_candidates):
        candidates.append(
            {
                "docid": f"doc{i}",
                "score": float(num_candidates - i),
                "doc": {"contents": f"passage text for document {i}"},
            }
        )
    return from_dict(
        data_class=Request,
        data={
            "query": {"text": query_text, "qid": qid},
            "candidates": candidates,
        },
    )


def _fake_rerank(query, documents, **kwargs):
    """Simulate Jina model.rerank(): return scores inversely proportional to index."""
    results = []
    for i, doc in enumerate(documents):
        results.append(
            {
                "index": i,
                "relevance_score": 1.0 / (i + 1),
                "document": doc,
            }
        )
    return results


def _build_reranker(batch_size=64, max_passage_words=None, context_size=131_072):
    """Build a JinaReranker with a mocked HuggingFace model."""
    mock_model = MagicMock()
    mock_model.rerank = MagicMock(side_effect=_fake_rerank)

    with patch("rank_llm.rerank.pointwise.jina_reranker.AutoModel") as mock_cls:
        mock_cls.from_pretrained.return_value = mock_model

        from rank_llm.rerank.pointwise.jina_reranker import JinaReranker

        reranker = JinaReranker(
            model="jinaai/jina-reranker-v3",
            device="cpu",
            batch_size=batch_size,
            max_passage_words=max_passage_words,
            context_size=context_size,
        )
    return reranker, mock_model


class TestExtractDocText(unittest.TestCase):
    def test_contents_field(self):
        from rank_llm.rerank.pointwise.jina_reranker import _extract_doc_text

        self.assertEqual(_extract_doc_text({"contents": "hello world"}), "hello world")

    def test_text_field(self):
        from rank_llm.rerank.pointwise.jina_reranker import _extract_doc_text

        self.assertEqual(_extract_doc_text({"text": "hello"}), "hello")

    def test_title_prepended(self):
        from rank_llm.rerank.pointwise.jina_reranker import _extract_doc_text

        result = _extract_doc_text({"title": "My Title", "contents": "body"})
        self.assertIn("My Title", result)
        self.assertIn("body", result)

    def test_max_words_truncation(self):
        from rank_llm.rerank.pointwise.jina_reranker import _extract_doc_text

        doc = {"text": "one two three four five six seven eight"}
        result = _extract_doc_text(doc, max_words=3)
        self.assertEqual(len(result.split()), 3)

    def test_no_truncation_when_none(self):
        from rank_llm.rerank.pointwise.jina_reranker import _extract_doc_text

        doc = {"text": "a b c d e"}
        result = _extract_doc_text(doc, max_words=None)
        self.assertEqual(result, "a b c d e")


class TestComputeEffectiveMaxWords(unittest.TestCase):
    def test_explicit_max_passage_words(self):
        reranker, _ = _build_reranker(max_passage_words=100)
        self.assertEqual(reranker._compute_effective_max_words(10), 100)

    def test_derived_max_passage_words(self):
        reranker, _ = _build_reranker(max_passage_words=None, context_size=1024)
        result = reranker._compute_effective_max_words(4)
        tokens_per_doc = (1024 - 128) // 4
        expected = int(tokens_per_doc * 0.75)
        self.assertEqual(result, expected)


class TestComputeDocsPerChunk(unittest.TestCase):
    def test_fits_many(self):
        reranker, _ = _build_reranker(batch_size=64, context_size=131_072)
        result = reranker._compute_docs_per_chunk(avg_doc_words=50)
        self.assertGreater(result, 0)
        self.assertLessEqual(result, 64)

    def test_large_docs_reduce_chunk(self):
        reranker, _ = _build_reranker(batch_size=64, context_size=4096)
        result = reranker._compute_docs_per_chunk(avg_doc_words=500)
        self.assertLess(result, 64)
        self.assertGreater(result, 0)


class TestJinaRerankerRerankBatch(unittest.TestCase):
    def test_basic_rerank(self):
        reranker, mock_model = _build_reranker()
        request = _make_request(num_candidates=5)
        results = reranker.rerank_batch([request])

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(len(result.candidates), 5)
        for i in range(len(result.candidates) - 1):
            self.assertGreaterEqual(
                result.candidates[i].score, result.candidates[i + 1].score
            )
        mock_model.rerank.assert_called_once()

    def test_chunking_calls(self):
        reranker, mock_model = _build_reranker(batch_size=3)
        request = _make_request(num_candidates=7)
        results = reranker.rerank_batch([request])

        result = results[0]
        self.assertEqual(len(result.candidates), 7)
        for i in range(len(result.candidates) - 1):
            self.assertGreaterEqual(
                result.candidates[i].score, result.candidates[i + 1].score
            )
        # 7 docs / batch_size 3 -> 3 calls (3 + 3 + 1)
        self.assertEqual(mock_model.rerank.call_count, 3)

    def test_raw_scores_used_across_chunks(self):
        """Scores are absolute (no normalisation) so raw values are kept."""
        reranker, _ = _build_reranker(batch_size=2)
        request = _make_request(num_candidates=4)
        results = reranker.rerank_batch([request])

        result = results[0]
        scores = [c.score for c in result.candidates]
        # _fake_rerank gives 1/1, 1/2 for first chunk and 1/1, 1/2 for second
        # Raw scores should be preserved: two candidates at 1.0 and two at 0.5
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 1.0)
        self.assertAlmostEqual(scores[2], 0.5)
        self.assertAlmostEqual(scores[3], 0.5)

    def test_scores_are_native_float(self):
        """Scores must be Python float, not numpy.float32, for JSON serialization."""
        reranker, _ = _build_reranker()
        request = _make_request(num_candidates=3)
        results = reranker.rerank_batch([request])
        for c in results[0].candidates:
            self.assertIsInstance(c.score, float)

    def test_multiple_requests(self):
        reranker, mock_model = _build_reranker()
        requests = [_make_request(num_candidates=3, qid=i) for i in range(4)]
        results = reranker.rerank_batch(requests)

        self.assertEqual(len(results), 4)
        for result in results:
            self.assertEqual(len(result.candidates), 3)

    def test_does_not_mutate_input(self):
        reranker, _ = _build_reranker()
        request = _make_request(num_candidates=3)
        original_scores = [c.score for c in request.candidates]
        original_order = [c.docid for c in request.candidates]

        reranker.rerank_batch([request])

        self.assertEqual([c.score for c in request.candidates], original_scores)
        self.assertEqual([c.docid for c in request.candidates], original_order)

    def test_invocations_history_populated(self):
        reranker, _ = _build_reranker(batch_size=3)
        request = _make_request(num_candidates=7)
        results = reranker.rerank_batch([request], populate_invocations_history=True)

        result = results[0]
        # 7 docs / batch 3 -> 3 chunks -> 3 invocations
        self.assertEqual(len(result.invocations_history), 3)
        for inv in result.invocations_history:
            self.assertIn("query=", inv.prompt)

    def test_invocations_history_empty_by_default(self):
        reranker, _ = _build_reranker()
        request = _make_request(num_candidates=3)
        results = reranker.rerank_batch([request])
        self.assertEqual(len(results[0].invocations_history), 0)

    def test_max_passage_words_applied(self):
        reranker, mock_model = _build_reranker(max_passage_words=2)
        request = _make_request(num_candidates=2)
        reranker.rerank_batch([request])

        call_args = mock_model.rerank.call_args
        docs = call_args[0][1]
        for doc in docs:
            self.assertLessEqual(len(doc.split()), 2)

    def test_single_candidate(self):
        reranker, _ = _build_reranker()
        request = _make_request(num_candidates=1)
        results = reranker.rerank_batch([request])
        self.assertEqual(len(results[0].candidates), 1)

    def test_abstract_methods_raise(self):
        reranker, _ = _build_reranker()
        with self.assertRaises(NotImplementedError):
            reranker.run_llm("test")
        with self.assertRaises(NotImplementedError):
            reranker.run_llm_batched(["test"])


class TestJinaRerankerCreatePrompt(unittest.TestCase):
    def test_create_prompt_format(self):
        reranker, _ = _build_reranker()
        request = _make_request(num_candidates=3)
        result_obj = Result(
            query=copy.deepcopy(request.query),
            candidates=copy.deepcopy(request.candidates),
            invocations_history=[],
        )
        prompt, count = reranker.create_prompt(result_obj, index=0)
        self.assertIn("Query:", prompt)
        self.assertIn("Document:", prompt)
        self.assertGreater(count, 0)


class TestJinaRerankerMisc(unittest.TestCase):
    def test_cost_is_zero(self):
        reranker, _ = _build_reranker()
        self.assertEqual(reranker.cost_per_1k_token(input_token=True), 0)
        self.assertEqual(reranker.cost_per_1k_token(input_token=False), 0)

    def test_num_output_tokens_is_zero(self):
        reranker, _ = _build_reranker()
        self.assertEqual(reranker.num_output_tokens(), 0)

    def test_get_num_tokens(self):
        reranker, _ = _build_reranker()
        tokens = reranker.get_num_tokens("hello world foo bar")
        self.assertGreater(tokens, 0)

    def test_batch_size_capped_at_64(self):
        reranker, _ = _build_reranker(batch_size=128)
        self.assertLessEqual(reranker._batch_size, 64)


if __name__ == "__main__":
    unittest.main()

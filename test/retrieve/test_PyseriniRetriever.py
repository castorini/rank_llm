import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from dacite import from_dict

from rank_llm.data import Request
from rank_llm.retrieve import INDICES, PyseriniRetriever, RetrievalMethod

valid_inputs = [
    ("dl19", RetrievalMethod.BM25),
    ("dl19", RetrievalMethod.BM25_RM3),
    ("dl20", RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL),
    ("dl20", RetrievalMethod.D_BERT_KD_TASB),
    ("dl20", RetrievalMethod.OPEN_AI_ADA2),
]

failure_inputs = [
    ("dl18", RetrievalMethod.BM25),  # dataset error
    ("dl18", RetrievalMethod.BM25_RM3),  # dataset error
    ("dl18", RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL),  # dataset error
    ("dl19", RetrievalMethod.UNSPECIFIED),  # retrieval method error
    ("dl16", RetrievalMethod.UNSPECIFIED),  # dataset and retrieval method error
    ("dl21", RetrievalMethod.D_BERT_KD_TASB),
    ("covid", RetrievalMethod.OPEN_AI_ADA2),
]


# Mocking Hits object
class MockHit:
    def __init__(self, docid: str, rank: int, score: float, qid: str) -> None:
        self.docid = docid
        self.rank = rank
        self.score = score
        self.qid = qid


class TestPyseriniRetriever(unittest.TestCase):
    def _make_retriever(
        self,
        dataset: str,
        retrieval_method: RetrievalMethod,
        *,
        topics: dict[str, dict[str, str]] | None = None,
    ) -> PyseriniRetriever:
        if topics is None:
            topics = {"q1": {"title": "Sample Query"}}

        with ExitStack() as stack:
            mock_lucene_searcher = stack.enter_context(
                patch("rank_llm.retrieve.pyserini_retriever.LuceneSearcher")
            )
            mock_impact_searcher = stack.enter_context(
                patch("rank_llm.retrieve.pyserini_retriever.LuceneImpactSearcher")
            )
            mock_faiss_searcher = stack.enter_context(
                patch("rank_llm.retrieve.pyserini_retriever.FaissSearcher")
            )
            mock_query_encoder = stack.enter_context(
                patch("rank_llm.retrieve.pyserini_retriever.QueryEncoder")
            )
            mock_index_reader = stack.enter_context(
                patch("rank_llm.retrieve.pyserini_retriever.LuceneIndexReader")
            )
            stack.enter_context(
                patch(
                    "rank_llm.retrieve.pyserini_retriever.get_topics",
                    return_value=topics,
                )
            )
            stack.enter_context(
                patch("rank_llm.retrieve.pyserini_retriever.get_qrels", return_value={})
            )

            mock_lucene_searcher.from_prebuilt_index.return_value = MagicMock()
            mock_impact_searcher.from_prebuilt_index.return_value = MagicMock()
            mock_faiss_searcher.from_prebuilt_index.return_value = MagicMock()
            mock_query_encoder.load_encoded_queries.return_value = MagicMock()
            mock_index_reader.from_prebuilt_index.return_value = MagicMock()

            return PyseriniRetriever(dataset, retrieval_method)

    def test_valid_inputs(self) -> None:
        for dataset, retrieval_method in valid_inputs:
            retriever = self._make_retriever(dataset, retrieval_method)
            self.assertEqual(retriever._dataset, dataset)
            self.assertEqual(retriever._retrieval_method, retrieval_method)
            self.assertIsNotNone(retriever._searcher)
            key = retrieval_method.value
            if key == "bm25_rm3":
                key = "bm25"
            self.assertEqual(retriever._get_index(), INDICES[key][dataset])

    def test_failure_inputs(self) -> None:
        for dataset, retrieval_method in failure_inputs:
            with self.subTest(dataset=dataset, retrieval_method=retrieval_method):
                with self.assertRaises(ValueError):
                    self._make_retriever(dataset, retrieval_method)

    def test_get_index(self) -> None:
        # Creating PyseriniRetriever instance
        retriever = self._make_retriever("dl19", RetrievalMethod.BM25)

        # Testing for a valid dataset
        index = retriever._get_index()
        self.assertEqual(index, "msmarco-v1-passage")

        # Testing for an invalid dataset
        with self.assertRaises(ValueError):
            retriever._dataset = "invalid_dataset"
            retriever._retrieval_method = RetrievalMethod.BM25_RM3
            retriever._get_index()

    @patch("rank_llm.retrieve.pyserini_retriever.json.loads")
    def test_retrieve_query(self, mock_json_loads: MagicMock) -> None:
        # Mocking json.loads to return a predefined content
        mock_json_loads.return_value = {"title": "Sample Title", "text": "Sample Text"}

        # Mocking hits
        mock_hits = MagicMock(spec=list[MockHit])
        mock_hits.__iter__.return_value = [
            MockHit("d1", 1, 0.5, "q1"),
            MockHit("d2", 2, 0.4, "q1"),
        ]
        # Setting up PyseriniRetriever instance
        retriever = self._make_retriever("dl19", RetrievalMethod.BM25)

        # Mocking the search method to return mock_hits
        retriever._searcher.search = MagicMock(return_value=mock_hits)
        retriever._index_reader.doc.return_value.raw.return_value = "{}"

        # Creating lists to store expected and actual results
        expected_results = [
            from_dict(
                data_class=Request,
                data={
                    "query": {"text": "Sample Query", "qid": "q1"},
                    "candidates": [
                        {
                            "doc": {"title": "Sample Title", "text": "Sample Text"},
                            "docid": "d1",
                            "score": 0.5,
                        },
                        {
                            "doc": {"title": "Sample Title", "text": "Sample Text"},
                            "docid": "d2",
                            "score": 0.4,
                        },
                    ],
                },
            )
        ]
        actual_results: list[Request] = []

        # Calling the _retrieve_query method
        retriever._retrieve_query("Sample Query", actual_results, 2, "q1")

        # Asserting that Hits object is called with the correct query and k
        retriever._searcher.search.assert_called_once_with("Sample Query", k=2)

        # Asserting the actual results match the expected results
        self.assertEqual(actual_results.__repr__(), expected_results.__repr__())

        # Reset the mocks for clean-up
        mock_json_loads.reset_mock()
        retriever._searcher.search.reset_mock()

    @patch("rank_llm.retrieve.pyserini_retriever.get_topics")
    def test_num_queries(self, mock_get_topics: MagicMock) -> None:
        # Mocking get_topics method to return a predefined number of queries
        mock_get_topics.return_value = {
            "query1": {"title": "Sample Title 1"},
            "query2": {"title": "Sample Title 2"},
            "query3": {"title": "Sample Title 3"},
        }

        # Creating PyseriniRetriever instance
        retriever = self._make_retriever(
            "dl19", RetrievalMethod.BM25, topics=mock_get_topics.return_value
        )

        # Asserting the number of queries
        self.assertEqual(retriever.num_queries(), 3)

        # Reset the mock for clean-up
        mock_get_topics.reset_mock()


if __name__ == "__main__":
    unittest.main()

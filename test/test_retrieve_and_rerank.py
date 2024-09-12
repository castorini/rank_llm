import unittest
from unittest.mock import MagicMock, patch

from rank_llm.retrieve_and_rerank import retrieve_and_rerank


# Anserini API must be hosted at 8081
class TestRetrieveAndRerank(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("rank_llm.rerank.listwise.rank_listwise_os_llm.load_model")
        self.mock_load_model = self.patcher.start()
        self.mock_llm = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_load_model.return_value = self.mock_llm, self.mock_tokenizer

        self.patcher_cuda = patch("torch.cuda.is_available")
        self.mock_cuda = self.patcher_cuda.start()
        self.mock_cuda.return_value = True

    def tearDown(self):
        self.patcher.stop()
        self.patcher_cuda.stop()

    def test_multiple_datasets(self):
        result = retrieve_and_rerank(
            "unspecified",
            query="chocolate",
            dataset=["msmarco-v2.1-doc", "msmarco-v2.1-doc"],
            interactive=True,
        )

        self.assertEqual(len(result[0]), 2)
        for x in result[0]:
            self.assertEqual(len(x.candidates), 10)

    def test_invalid_retrieval_mode(self):
        # Test handling of invalid retrieval modes
        with self.assertRaises(TypeError):
            retrieve_and_rerank(
                "rank_zephyr",
                ["msmarco-v2.1-doc"],
                interactive=True,
                retrieval_mode="INVALID_MODE",
            )

    def test_improper_arguments(self):
        # Missing dataset
        with self.assertRaises(TypeError):
            retrieve_and_rerank(
                "unspecified", query="hi", interactive=True, num_passes=4
            )
        # Missing model path
        with self.assertRaises(TypeError):
            retrieve_and_rerank(
                query="hi", dataset=["msmarco-v2.1-doc"], interactive=True, num_passes=4
            )
        # Missing query
        with self.assertRaises(TypeError):
            retrieve_and_rerank(
                "rank_zephyr",
                dataset=["msmarco-v2.1-doc"],
                interactive=True,
                num_passes=4,
            )

    def test_multiple_passes(self):
        patcher_rerank_batch = patch(
            "rank_llm.rerank.listwise.RankListwiseOSLLM.rerank_batch", return_value=[]
        )
        mock_rerank_batch = patcher_rerank_batch.start()
        # Test reranking with multiple passes
        retrieve_and_rerank(
            "rank_zephyr",
            query="hi",
            dataset=["msmarco-v2.1-doc"],
            interactive=True,
            num_passes=4,
        )
        # Ensure rerank_batch is called 4 times
        self.assertEqual(mock_rerank_batch.call_count, 4)

        patcher_rerank_batch.stop()

    def test_identity_reranker(self):
        result = retrieve_and_rerank(
            "rank_random",
            query="what is canada",
            dataset=["msmarco-v2.1-doc"],
            top_k_rerank=6,
            interactive=True,
        )
        self.assertEqual(len(result[0][0].candidates), 6)

    def test_identity_reranker_called(self):
        patcher_rerank_batch = patch(
            "rank_llm.rerank.IdentityReranker.rerank_batch", return_value=[]
        )

        for model in ["rank_random", "unspecified"]:
            mock_rerank_batch = patcher_rerank_batch.start()

            retrieve_and_rerank(
                model_path=model,
                query="sample query",
                dataset="msmarco-v2.1-doc",
                top_k_rerank=6,
                interactive=True,
            )
            self.assertTrue(
                mock_rerank_batch.called,
                "IdentityReranker's rerank_batch was not called",
            )

            patcher_rerank_batch.stop()


if __name__ == "__main__":
    unittest.main()

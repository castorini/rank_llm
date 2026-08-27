import unittest
from unittest.mock import MagicMock

from rank_llm.rerank.listwise.vicuna_reranker import VicunaReranker
from rank_llm.rerank.listwise.zephyr_reranker import ZephyrReranker


class TestListwiseRerankerCleanup(unittest.TestCase):
    def test_convenience_rerankers_delegate_close(self):
        for reranker_class in (ZephyrReranker, VicunaReranker):
            with self.subTest(reranker=reranker_class.__name__):
                reranker = reranker_class.__new__(reranker_class)
                reranker._reranker = MagicMock()

                reranker.close()

                reranker._reranker.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

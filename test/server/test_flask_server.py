import unittest
from contextlib import ExitStack
from typing import Any, cast
from unittest.mock import MagicMock, patch

from rank_llm.server.flask.api import create_app

# Needs Anserini API to be active at 8081

# - name: Run API tests
#   run: |
#     python -m unittest discover -s test/api
# - name: Run retrieve_and_rerank
#   run: |
#     python -m unittest test/test_retrieve_and_rerank


class TestAPI(unittest.TestCase):
    BASE_URL = "http://localhost:8082/api/model/{model_name}/index/{index_name}/{anserini_host_addr}"

    def setUp(self) -> None:
        self._patches = ExitStack()
        self.mock_cuda = self._patches.enter_context(patch("torch.cuda.is_available"))
        self.mock_cuda.return_value = True
        self.mock_rank_listwise = self._patches.enter_context(
            patch("rank_llm.rerank.listwise.RankListwiseOSLLM")
        )
        self.mock_identity = self._patches.enter_context(
            patch("rank_llm.rerank.IdentityReranker")
        )
        self.mock_empty_cache = self._patches.enter_context(
            patch("torch.cuda.empty_cache")
        )
        self.mock_retrieve_and_rerank = self._patches.enter_context(
            patch("rank_llm.retrieve_and_rerank.retrieve_and_rerank")
        )

        self.default_model = MagicMock()
        self.default_model.get_name.return_value = "rank_zephyr"
        self.mock_rank_listwise.return_value = self.default_model

        def retrieve_side_effect(
            *args: Any, **kwargs: Any
        ) -> tuple[list[dict[str, Any]], Any]:
            if not kwargs["query"]:
                raise ValueError("query is required")
            model_name = kwargs["model_path"]
            coordinator = MagicMock()
            coordinator.get_name.return_value = model_name
            return (
                [
                    {
                        "query": {"text": kwargs["query"], "qid": str(kwargs["qid"])},
                        "candidates": [
                            {
                                "docid": f"d{i}",
                                "score": float(10 - i),
                                "doc": {"text": f"doc {i}"},
                            }
                            for i in range(kwargs["top_k_rerank"])
                        ],
                        "invocations_history": [],
                    }
                ],
                coordinator,
            )

        self.mock_retrieve_and_rerank.side_effect = retrieve_side_effect

        # Mock RankLLM host at port 8082
        self.app, _ = create_app("rank_zephyr", 8082, False)
        self.client = self.app.test_client()

        # Define commonly used API parameters
        self.model_name = "rank_zephyr"
        self.index_name = "msmarco-v2.1-doc"
        self.anserini_host_addr = "8081"
        self.query = "Who killed the Yardbirds"
        self.hits_retriever = 10
        self.hits_reranker = 4
        self.qid = 1
        self.num_passes = 1

        # Request query parameters
        self.query_params = {
            "query": self.query,
            "hits_retriever": self.hits_retriever,
            "hits_reranker": self.hits_reranker,
            "qid": self.qid,
            "num_passes": self.num_passes,
        }

    def tearDown(self) -> None:
        self._patches.close()

    def test_basic_response_structure(self) -> None:
        """Test that the API returns a valid JSON and status code 200 for a correct request."""
        response = self.client.get(
            self.BASE_URL.format(
                model_name=self.model_name,
                index_name=self.index_name,
                anserini_host_addr=self.anserini_host_addr,
            ),
            query_string=self.query_params,
        )
        self.assertEqual(response.status_code, 200)
        payload = cast(dict[str, Any], response.get_json())
        self.assertIsInstance(payload, dict)
        self.assertEqual(len(payload), 3)
        self.assertEqual(len(payload["candidates"]), 4)

    def test_optional_parameters(self) -> None:
        """Test that the API correctly uses default values for optional parameters."""
        # Removing 'hits_retriever' and 'hits_reranker' to test default values
        query_params = self.query_params.copy()
        query_params.pop("hits_retriever")
        query_params.pop("hits_reranker")

        response = self.client.get(
            self.BASE_URL.format(
                model_name=self.model_name,
                index_name=self.index_name,
                anserini_host_addr=self.anserini_host_addr,
            ),
            query_string=query_params,
        )
        self.assertEqual(response.status_code, 200)
        payload = cast(dict[str, Any], response.get_json())
        self.assertIsInstance(payload, dict)
        self.assertEqual(len(payload["candidates"]), 10)

    def test_missing_query_parameter(self) -> None:
        """Test that the API handles missing 'query' parameter gracefully."""
        query_params = self.query_params.copy()
        query_params.pop("query")

        response = self.client.get(
            self.BASE_URL.format(
                model_name=self.model_name,
                index_name=self.index_name,
                anserini_host_addr=self.anserini_host_addr,
            ),
            query_string=query_params,
        )
        self.assertEqual(response.status_code, 500)

    def test_invalid_retrieval_method(self) -> None:
        """Test that the server handles unsupported retrieval methods properly."""
        query_params = self.query_params.copy()
        query_params["retrieval_method"] = "invalid_method"

        response = self.client.get(
            self.BASE_URL.format(
                model_name=self.model_name,
                index_name=self.index_name,
                anserini_host_addr=self.anserini_host_addr,
            ),
            query_string=query_params,
        )
        self.assertEqual(response.status_code, 500)
        payload = cast(dict[str, Any], response.get_json())
        self.assertIn("error", payload)

    def test_model_caching(self) -> None:
        """Test that server caching works"""
        model_names = [
            "rank_zephyr",
            "unspecified",
            "rank_zephyr",
            "rank_vicuna",
            "rank_vicuna",
            "unspecified",
            "unspecified",
        ]

        for model_name in model_names:
            response = self.client.get(
                self.BASE_URL.format(
                    model_name=model_name,
                    index_name=self.index_name,
                    anserini_host_addr=self.anserini_host_addr,
                ),
                query_string=self.query_params,
            )
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.get_json(), dict)
        self.mock_empty_cache.assert_called()


if __name__ == "__main__":
    unittest.main()

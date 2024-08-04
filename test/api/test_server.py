import unittest

import requests

# Needs RankLLM API to be active at 8082 and Anserini API to be active at 8081
# If not mocked, this test suite takes around 5min on a single RTX4090

# - name: Run API tests
#   run: |
#     python -m unittest discover -s test/api
# - name: Run retrieve_and_rerank
#   run: |
#     python -m unittest test/test_retrieve_and_rerank


class TestAPI(unittest.TestCase):
    BASE_URL = "http://localhost:8082/api/model/{model_name}/index/{index_name}/{anserini_host_addr}"

    def setUp(self):
        self.model_name = "rank_zephyr"
        self.index_name = "msmarco-v2.1-doc"
        self.anserini_host_addr = "8081"
        self.query = "Who%20are%20the%20Yardbirds"
        self.hits_retriever = 10
        self.hits_reranker = 4
        self.qid = 1
        self.num_passes = 1

        self.query_params = {
            "query": self.query,
            "hits_retriever": self.hits_retriever,
            "hits_reranker": self.hits_reranker,
            "qid": self.qid,
            "num_passes": self.num_passes,
        }

    def test_basic_response_structure(self):
        """Test that the API returns a valid JSON and status code 200 for a correct request."""
        response = requests.get(
            self.BASE_URL.format(
                model_name=self.model_name,
                index_name=self.index_name,
                anserini_host_addr=self.anserini_host_addr,
            ),
            params=self.query_params,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), dict)

    def test_optional_parameters(self):
        """Test that the API correctly uses default values for optional parameters."""
        # Removing 'hits_retriever' and 'hits_reranker' to test default values
        query_params = self.query_params.copy()
        query_params.pop("hits_retriever")
        query_params.pop("hits_reranker")

        response = requests.get(
            self.BASE_URL.format(
                model_name=self.model_name,
                index_name=self.index_name,
                anserini_host_addr=self.anserini_host_addr,
            ),
            params=query_params,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), dict)

    def test_missing_query_parameter(self):
        """Test that the API handles missing 'query' parameter gracefully."""
        query_params = self.query_params.copy()
        query_params.pop("query")

        response = requests.get(
            self.BASE_URL.format(
                model_name=self.model_name,
                index_name=self.index_name,
                anserini_host_addr=self.anserini_host_addr,
            ),
            params=query_params,
        )
        self.assertNotEqual(response.status_code, 200)

    def test_invalid_retrieval_method(self):
        """Test that the server handles unsupported retrieval methods properly."""
        query_params = self.query_params.copy()
        query_params["retrieval_method"] = "invalid_method"

        response = requests.get(
            self.BASE_URL.format(
                model_name=self.model_name,
                index_name=self.index_name,
                anserini_host_addr=self.anserini_host_addr,
            ),
            params=query_params,
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json())

    def test_model_caching(self):
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
            response = requests.get(
                self.BASE_URL.format(
                    model_name=model_name,
                    index_name=self.index_name,
                    anserini_host_addr=self.anserini_host_addr,
                ),
                params=self.query_params,
            )
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.json(), dict)


if __name__ == "__main__":
    unittest.main()

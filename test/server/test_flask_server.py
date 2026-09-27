import unittest
from unittest.mock import MagicMock, patch

from rank_llm.server.flask.api import create_app


class TestAPI(unittest.TestCase):
    BASE_URL = "http://localhost:8082/api/model/{model_name}/index/{index_name}/{anserini_host_addr}"

    def setUp(self):
        model_class = self.enterContext(
            patch("rank_llm.rerank.listwise.RankListwiseOSLLM")
        )
        model_class.return_value.get_name.return_value = "rank_zephyr"
        self.mock_retrieve_and_rerank = self.enterContext(
            patch("rank_llm.retrieve_and_rerank.retrieve_and_rerank")
        )
        self.mock_empty_cache = self.enterContext(patch("torch.cuda.empty_cache"))

        def retrieve(*, model_path, query, top_k_rerank, **kwargs):
            coordinator = MagicMock()
            coordinator.get_name.return_value = model_path
            result = {
                "query": {"text": query, "qid": kwargs["qid"]},
                "candidates": [
                    {"docid": str(i), "score": 1.0, "doc": {"contents": "text"}}
                    for i in range(top_k_rerank)
                ],
                "invocations_history": [],
            }
            return [result], coordinator

        self.mock_retrieve_and_rerank.side_effect = retrieve
        self.app, _ = create_app("rank_zephyr", 8082, False)
        self.client = self.app.test_client()
        self.model_name = "rank_zephyr"
        self.index_name = "msmarco-v2.1-doc"
        self.anserini_host_addr = "8081"
        self.query_params = {
            "query": "Who killed the Yardbirds",
            "hits_retriever": 10,
            "hits_reranker": 4,
            "qid": 1,
            "num_passes": 1,
        }

    def _get(self, model_name=None, query_params=None):
        return self.client.get(
            self.BASE_URL.format(
                model_name=model_name or self.model_name,
                index_name=self.index_name,
                anserini_host_addr=self.anserini_host_addr,
            ),
            query_string=query_params
            if query_params is not None
            else self.query_params,
        )

    def test_basic_response_structure(self):
        response = self._get()
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json, dict)
        self.assertEqual(len(response.json["candidates"]), 4)
        self.assertEqual(response.json["query"]["text"], self.query_params["query"])
        self.assertEqual(
            self.mock_retrieve_and_rerank.call_args.kwargs["host"],
            "http://localhost:8081",
        )

    def test_optional_parameters(self):
        query_params = self.query_params.copy()
        query_params.pop("hits_retriever")
        query_params.pop("hits_reranker")

        response = self._get(query_params=query_params)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json["candidates"]), 10)
        self.assertEqual(
            self.mock_retrieve_and_rerank.call_args.kwargs["top_k_retrieve"], 20
        )

    def test_missing_query_is_forwarded_to_backend(self):
        query_params = self.query_params.copy()
        query_params.pop("query")

        response = self._get(query_params=query_params)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.mock_retrieve_and_rerank.call_args.kwargs["query"])

    def test_downstream_error_is_json(self):
        self.mock_retrieve_and_rerank.side_effect = ValueError("retrieval failed")

        response = self._get()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {"error": "retrieval failed"})

    def test_invalid_retrieval_method(self):
        query_params = self.query_params.copy()
        query_params["retrieval_method"] = "invalid_method"

        response = self._get(query_params=query_params)
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json)
        self.mock_retrieve_and_rerank.assert_not_called()

    def test_model_caching(self):
        for model_name in ("rank_zephyr", "rank_zephyr", "rank_vicuna", "rank_vicuna"):
            response = self._get(model_name=model_name)
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.mock_empty_cache.call_count, 1)


if __name__ == "__main__":
    unittest.main()

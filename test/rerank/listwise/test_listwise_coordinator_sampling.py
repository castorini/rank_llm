"""Sampling overrides must survive the listwise coordinator factory.

RankListwiseOSLLM accepts sampling_kwargs, but the factory used by
`listwise_vllm_with_openai_sdk` previously dropped it, so a model with a
prescribed decoding configuration (RubricRanker: temperature 0.7, top_p 0.8,
top_k 20, min_p 0) silently got the defaults instead.
"""

import unittest
from unittest.mock import patch

from rank_llm.rerank.reranker import _create_rank_listwise_os_llm_coordinator

SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


class TestListwiseCoordinatorSampling(unittest.TestCase):
    def _build(self, **kwargs):
        with patch("rank_llm.rerank.listwise.RankListwiseOSLLM") as mock:
            _create_rank_listwise_os_llm_coordinator(
                "some/model", base_url="http://localhost:8000/v1", **kwargs
            )
            self.assertTrue(mock.called, "coordinator was not constructed")
            return mock.call_args.kwargs

    def test_sampling_kwargs_are_forwarded(self):
        self.assertEqual(
            self._build(sampling_kwargs=SAMPLING)["sampling_kwargs"], SAMPLING
        )

    def test_absent_sampling_kwargs_stays_none(self):
        self.assertIsNone(self._build()["sampling_kwargs"])

    def test_base_url_still_forwarded(self):
        self.assertEqual(
            self._build(sampling_kwargs=SAMPLING)["base_url"],
            "http://localhost:8000/v1",
        )


if __name__ == "__main__":
    unittest.main()

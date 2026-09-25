import unittest
from unittest.mock import Mock

from rank_llm.api.operations import (
    run_retrieve_and_rerank,
)
from rank_llm.api.options import RerankOptions, RetrievalOptions
from rank_llm.retrieve import RetrievalMethod, RetrievalMode


class TestCLIOperations(unittest.TestCase):
    def test_run_retrieve_and_rerank_normalizes_inputs(self):
        runner = Mock(return_value=["results"])
        result = run_retrieve_and_rerank(
            options=RerankOptions(
                model_path="model",
                prompt_template_path="prompt.yaml",
                few_shot_file="few.jsonl",
                base_url="https://example.invalid/v1",
                reasoning_effort="high",
                max_passage_words=222,
                use_litellm=True,
            ),
            retrieval=RetrievalOptions(
                dataset="dl19",
                retrieval_method=RetrievalMethod.BM25,
                max_queries=-1,
            ),
            runner=runner,
            device_resolver=lambda: "cpu",
        )

        self.assertEqual(result, ["results"])
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["retrieval_mode"], RetrievalMode.DATASET)
        self.assertEqual(kwargs["top_k_rerank"], 100)
        self.assertEqual(kwargs["prompt_template_path"], "prompt.yaml")
        self.assertEqual(kwargs["few_shot_file"], "few.jsonl")
        self.assertEqual(kwargs["base_url"], "https://example.invalid/v1")
        self.assertIsNone(kwargs["max_queries"])
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["max_passage_words"], 222)
        self.assertTrue(kwargs["use_litellm"])


if __name__ == "__main__":
    unittest.main()

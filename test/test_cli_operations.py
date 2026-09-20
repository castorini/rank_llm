import unittest
from unittest.mock import Mock

from rank_llm.api.operations import (
    run_evaluate_aggregate,
    run_response_analysis_files,
    run_retrieve_and_rerank,
    run_retrieve_cache_generation,
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

    def test_run_evaluate_aggregate_uses_custom_runner(self):
        runner = Mock()
        summary = run_evaluate_aggregate(
            model_name="model",
            context_size=2048,
            rerank_results_dirname="results",
            runner=runner,
        )

        runner.assert_called_once_with("model", 2048, "results")
        self.assertEqual(
            summary["output_file"], "trec_eval_aggregated_results_model.jsonl"
        )

    def test_run_response_analysis_files_uses_custom_runner(self):
        runner = Mock(return_value={"files": ["one.json"], "metrics": {"errors": 0}})
        summary = run_response_analysis_files(
            files=["one.json"],
            verbose=True,
            runner=runner,
        )

        runner.assert_called_once_with(["one.json"], True)
        self.assertEqual(summary["metrics"]["errors"], 0)

    def test_run_retrieve_cache_generation_uses_generator_and_writer(self):
        generator = Mock(return_value=[{"query": "cats"}])
        writer = Mock()

        summary = run_retrieve_cache_generation(
            trec_file="run.trec",
            collection_file="collection.tsv",
            query_file="queries.tsv",
            output_file="cache.json",
            output_trec_file="cache.trec",
            topk=10,
            generator=generator,
            writer=writer,
        )

        generator.assert_called_once_with(
            "run.trec",
            "collection.tsv",
            "queries.tsv",
            10,
            "cache.trec",
        )
        writer.assert_called_once_with("cache.json", [{"query": "cats"}])
        self.assertEqual(summary["record_count"], 1)


if __name__ == "__main__":
    unittest.main()

import argparse
import unittest
from unittest.mock import Mock

from rank_llm.cli.operations import (
    run_evaluate_aggregate,
    run_mcp_retrieve_and_rerank,
    run_response_analysis_files,
    run_retrieve_cache_generation,
    run_script_rerank,
)
from rank_llm.retrieve import RetrievalMethod, RetrievalMode


class TestCLIOperations(unittest.TestCase):
    def test_run_script_rerank_builds_expected_runner_args(self):
        args = argparse.Namespace(
            model_path="model",
            batch_size=4,
            use_azure_openai=False,
            use_openrouter=True,
            base_url="https://example.invalid/v1",
            context_size=2048,
            top_k_candidates=50,
            top_k_rerank=-1,
            max_queries=2,
            dataset="dl19",
            retrieval_method=RetrievalMethod.BM25,
            requests_file=None,
            qrels_file="qrels.txt",
            output_jsonl_file="out.jsonl",
            output_trec_file="out.trec",
            invocations_history_file="history.json",
            num_gpus=2,
            prompt_mode=None,
            prompt_template_path="template.yaml",
            shuffle_candidates=True,
            print_prompts_responses=False,
            num_few_shot_examples=1,
            few_shot_file="few.jsonl",
            variable_passages=True,
            num_passes=3,
            window_size=20,
            stride=10,
            system_message="system",
            populate_invocations_history=True,
            is_thinking=True,
            reasoning_token_budget=200,
            use_logits=True,
            use_alpha=True,
            sglang_batched=False,
            tensorrt_batched=True,
            reasoning_effort="medium",
            max_passage_words=123,
        )
        runner = Mock(return_value=["ok"])
        result = run_script_rerank(
            args,
            parser_error=Mock(),
            runner=runner,
            device_resolver=lambda: "cpu",
        )

        self.assertEqual(result.results, ["ok"])
        runner.assert_called_once()
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["retrieval_mode"], RetrievalMode.DATASET)
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["top_k_rerank"], 50)
        self.assertEqual(str(kwargs["prompt_template_path"]), "template.yaml")

    def test_run_script_rerank_uses_parser_error_for_invalid_combo(self):
        args = argparse.Namespace(
            dataset=None,
            requests_file="requests.jsonl",
            retrieval_method=RetrievalMethod.BM25,
        )
        parser_error = Mock(side_effect=RuntimeError("bad args"))
        with self.assertRaisesRegex(RuntimeError, "bad args"):
            run_script_rerank(args, parser_error=parser_error)
        parser_error.assert_called_once_with(
            "--retrieval_method must not be used with --requests_file"
        )

    def test_run_mcp_retrieve_and_rerank_normalizes_inputs(self):
        runner = Mock(return_value=["results"])
        result = run_mcp_retrieve_and_rerank(
            model_path="model",
            dataset="dl19",
            retrieval_method=RetrievalMethod.BM25,
            prompt_template_path="prompt.yaml",
            few_shot_file="few.jsonl",
            base_url="https://example.invalid/v1",
            max_queries=-1,
            reasoning_effort="high",
            max_passage_words=222,
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

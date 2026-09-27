import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import DataWriter, Query, Result
from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM


class TestInvocationHistory(unittest.TestCase):
    def setUp(self):
        with patch.object(ListwiseRankLLM, "__abstractmethods__", frozenset()):
            self.reranker = object.__new__(ListwiseRankLLM)
        self.reranker._inference_handler = MagicMock()
        self.reranker._inference_handler.template = {
            "output_validation_regex": r'r"^\[\d+\]( > \[\d+\])*$"',
            "output_extraction_regex": r'r"\[(\d+)\]"',
        }
        self.reranker.receive_permutation = MagicMock(
            side_effect=lambda result, *_: result
        )
        self.result = Result(query=Query(text="query", qid="q1"))

    def assert_legacy_history(self):
        self.assertEqual(len(self.result.invocations_history), 1)
        invocation = self.result.invocations_history[0]
        self.assertEqual(invocation.prompt, "[1] [2]")
        self.assertEqual(invocation.response, "[1] > [2]")
        self.assertEqual(invocation.input_token_count, 3)
        self.assertEqual(invocation.output_token_count, 2)
        self.assertIsNone(invocation.reasoning)
        self.assertIsNone(invocation.token_usage)
        self.assertEqual(
            invocation.output_validation_regex,
            self.reranker._inference_handler.template["output_validation_regex"],
        )
        self.assertEqual(
            invocation.output_extraction_regex,
            self.reranker._inference_handler.template["output_extraction_regex"],
        )
        self.assertEqual(
            ResponseAnalyzer.from_inline_results([self.result]).read_responses(),
            (
                ["[1] > [2]"],
                [2],
                invocation.output_validation_regex,
                invocation.output_extraction_regex,
            ),
        )

    def test_default_history_is_independent_empty_list_and_serializable(self):
        another_result = Result(query=Query(text="another", qid="q2"))
        self.assertEqual(self.result.invocations_history, [])
        self.assertIsNot(
            self.result.invocations_history, another_result.invocations_history
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "history.json"
            DataWriter(self.result).write_inference_invocations_history(str(output))
            self.assertEqual(
                json.loads(output.read_text()),
                [{"query": {"text": "query", "qid": "q1"}, "invocations_history": []}],
            )

    def test_async_pipeline_legacy_response_records_regex_fields(self):
        returned = self.reranker._apply_llm_output_to_result(
            self.result,
            ("[1] > [2]", 2),
            "[1] [2]",
            3,
            0,
            2,
            populate_invocations_history=True,
        )
        self.assertIs(returned, self.result)
        self.assert_legacy_history()

    def test_sync_pipeline_legacy_response_records_regex_fields(self):
        self.reranker.create_prompt = MagicMock(return_value=("[1] [2]", 3))
        self.reranker.run_llm = MagicMock(return_value=("[1] > [2]", 2))

        returned = self.reranker.permutation_pipeline(self.result, 0, 2)

        self.assertIs(returned, self.result)
        self.assert_legacy_history()


if __name__ == "__main__":
    unittest.main()

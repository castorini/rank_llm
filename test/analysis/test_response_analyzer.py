import unittest

from src.rank_llm.analysis.response_analysis import ResponseAnalyzer
from src.rank_llm.data import InferenceInvocation, Result


class TestResponseAnalyzer(unittest.TestCase):
    # create a list of mock Result objects
    def setUp(self):
        self.mock_results = [
            Result(
                query="Query 1",
                candidates=[],
                invocations_history=[
                    InferenceInvocation(
                        prompt="I will provide you with 3 passages: [1]Test, [2]Test, [3]Test",
                        response="[1] > [2] > [3]",
                        input_token_count=100,
                        output_token_count=50,
                        output_validation_regex='r"^\[\d+\]( > \[\d+\])*$"',
                        output_extraction_regex='r"\[(\d+)\]"',
                    ),
                    InferenceInvocation(
                        prompt="I will provide you with 2 passages: [1]Test, [2]Test",
                        response="[2] > [1]",
                        input_token_count=80,
                        output_token_count=40,
                        output_validation_regex='r"^\[\d+\]( > \[\d+\])*$"',
                        output_extraction_regex='r"\[(\d+)\]"',
                    ),
                ],
            ),
            Result(
                query="Query 2",
                candidates=[],
                invocations_history=[
                    InferenceInvocation(
                        prompt="I will provide you with 4 passages: [1]Test, [2]Test, [3]Test, [4]Test",
                        response="[4] > [3] > [2] > [1]",
                        input_token_count=120,
                        output_token_count=60,
                        output_validation_regex='r"^\[\d+\]( > \[\d+\])*$"',
                        output_extraction_regex='r"\[(\d+)\]"',
                    )
                ],
            ),
        ]

    def test_read_results_responses(self):
        analyzer = ResponseAnalyzer.from_inline_results(self.mock_results)
        (
            responses,
            num_passages,
            output_validation_regex,
            output_extraction_regex,
        ) = analyzer.read_results_responses()

        self.assertEqual(len(responses), 3, "Should have 3 responses")
        self.assertEqual(len(num_passages), 3, "Should have 3 num_passages")
        self.assertEqual(
            num_passages, [3, 2, 4], "Num passages should match expected values"
        )
        self.assertEqual(
            output_validation_regex,
            self.mock_results[0].invocations_history[0].output_validation_regex,
        )
        self.assertEqual(
            output_extraction_regex,
            self.mock_results[0].invocations_history[0].output_extraction_regex,
        )


if __name__ == "__main__":
    unittest.main()

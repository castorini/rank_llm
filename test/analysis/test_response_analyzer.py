import unittest

from src.rank_llm.analysis.response_analysis import ResponseAnalyzer
from src.rank_llm.data import InferenceInvocation, Result


class TestResponseAnalyzer(unittest.TestCase):
    # create a list of mock Result objects
    def setUp(self):
        output_patterns = [r"^\[\d+\]( > \[\d+\])*$", r"\[(\d+)\]"]
        self.mock_results = [
            Result(
                query="Query 1",
                candidates=[],
                invocations_history=[
                    InferenceInvocation(
                        prompt="I will provide you with 3 passages",
                        response="[1] > [2] > [3]",
                        input_token_count=100,
                        output_token_count=50,
                        output_patterns=output_patterns,
                    ),
                    InferenceInvocation(
                        prompt="I will provide you with 2 passages",
                        response="[2] > [1]",
                        input_token_count=80,
                        output_token_count=40,
                        output_patterns=output_patterns,
                    ),
                ],
            ),
            Result(
                query="Query 2",
                candidates=[],
                invocations_history=[
                    InferenceInvocation(
                        prompt="I will provide you with 4 passages",
                        response="[4] > [3] > [2] > [1]",
                        input_token_count=120,
                        output_token_count=60,
                        output_patterns=output_patterns,
                    )
                ],
            ),
        ]

    def test_read_results_responses(self):
        analyzer = ResponseAnalyzer.from_inline_results(self.mock_results)
        responses, num_passages, output_patterns = analyzer.read_results_responses()

        self.assertEqual(len(responses), 3, "Should have 3 responses")
        self.assertEqual(len(num_passages), 3, "Should have 3 num_passages")
        self.assertEqual(
            num_passages, [3, 2, 4], "Num passages should match expected values"
        )


if __name__ == "__main__":
    unittest.main()

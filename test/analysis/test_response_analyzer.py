import unittest

from src.rank_llm.analysis.response_analysis import ResponseAnalyzer
from src.rank_llm.data import RankingExecInfo, Result


class TestResponseAnalyzer(unittest.TestCase):
    # create a list of mock Result objects
    def setUp(self):
        self.mock_results = [
            Result(
                query="Query 1",
                candidates=[],
                ranking_exec_summary=[
                    RankingExecInfo(
                        prompt="I will provide you with 3 passages",
                        response="1 > 2 > 3",
                        input_token_count=100,
                        output_token_count=50,
                    ),
                    RankingExecInfo(
                        prompt="I will provide you with 2 passages",
                        response="2 > 1",
                        input_token_count=80,
                        output_token_count=40,
                    ),
                ],
            ),
            Result(
                query="Query 2",
                candidates=[],
                ranking_exec_summary=[
                    RankingExecInfo(
                        prompt="I will provide you with 4 passages",
                        response="4 > 3 > 2 > 1",
                        input_token_count=120,
                        output_token_count=60,
                    )
                ],
            ),
        ]

    def test_read_results_responses(self):
        analyzer = ResponseAnalyzer.from_inline_results(self.mock_results)
        responses, num_passages = analyzer.read_results_responses()

        self.assertEqual(len(responses), 3, "Should have 3 responses")
        self.assertEqual(len(num_passages), 3, "Should have 3 num_passages")
        self.assertEqual(
            num_passages, [3, 2, 4], "Num passages should match expected values"
        )


if __name__ == "__main__":
    unittest.main()

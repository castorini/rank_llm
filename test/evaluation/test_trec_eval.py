import unittest
from unittest.mock import patch

from src.rank_llm.evaluation.trec_eval import EvalFunction
from src.rank_llm.result import Result


class TestEvalFunction(unittest.TestCase):
    def setUp(self):
        self.results = [
            Result(
                query="Query1",
                hits=[
                    {"qid": "q1", "docid": "D1", "rank": 1, "score": 0.9},
                    {"qid": "q1", "docid": "D2", "rank": 2, "score": 0.8},
                ],
            ),
            Result(
                query="Query2",
                hits=[{"qid": "q2", "docid": "D3", "rank": 1, "score": 0.85}],
            ),
        ]
        self.qrels_path = "path/to/qrels"

    @patch("src.rank_llm.evaluation.trec_eval.EvalFunction.eval")
    def test_from_results(self, mock_eval):
        mock_eval.return_value = "Evaluation success"
        eval_output = EvalFunction.from_results(self.results, self.qrels_path)

        mock_eval.assert_called()
        self.assertEqual(eval_output, "Evaluation success")


if __name__ == "__main__":
    unittest.main()

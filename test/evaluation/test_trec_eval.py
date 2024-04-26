import unittest
from unittest.mock import patch

from dacite import from_dict

from src.rank_llm.evaluation.trec_eval import EvalFunction
from src.rank_llm.data import Result


class TestEvalFunction(unittest.TestCase):
    def setUp(self):
        self.results = [
            from_dict(
                data_class=Result,
                data={
                    "query": {"text": "Query1", "qid": "q1"},
                    "candidates": [
                        {"doc": {"text": "Doc1"}, "docid": "D1", "score": 0.9},
                        {"doc": {"text": "Doc2"}, "docid": "D2", "score": 0.8},
                    ],
                },
            ),
            from_dict(
                data_class=Result,
                data={
                    "query": {"text": "Query2", "qid": "q2"},
                    "candidates": [
                        {"doc": {"text": "Doc3"}, "docid": "D3", "score": 0.85}
                    ],
                },
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

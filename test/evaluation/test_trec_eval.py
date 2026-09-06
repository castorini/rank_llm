import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dacite import from_dict

from src.rank_llm.data import Result
from src.rank_llm.evaluation.trec_eval import EvalFunction


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


class TestEvalCommandArgs(unittest.TestCase):
    """The command handed to trec_eval must keep every option the caller passed."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        self.qrels = root / "qrels.txt"
        self.run = root / "run.txt"
        self.qrels.write_text("q1 0 D1 1\n", encoding="utf-8")
        self.run.write_text("q1 Q0 D1 1 0.9 rank_llm\n", encoding="utf-8")

    def _captured_cmd(self, call):
        """Run ``call`` with trec_eval stubbed out and return the command it built."""
        with (
            patch(
                "src.rank_llm.evaluation.trec_eval.download_evaluation_script",
                return_value="trec_eval.jar",
            ),
            patch(
                "src.rank_llm.evaluation.trec_eval._require_pandas",
                return_value=MagicMock(),
            ),
            patch(
                "src.rank_llm.evaluation.trec_eval.EvalFunction.trunc",
                staticmethod(lambda qrels, run: qrels),
            ),
            patch("src.rank_llm.evaluation.trec_eval.subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value.communicate.return_value = (b"ndcg 0.5", b"")
            call()
        return mock_popen.call_args[0][0]

    def _run_eval(self, args):
        return self._captured_cmd(lambda: EvalFunction.eval(args, trunc=False))

    def test_eval_keeps_first_option(self):
        cmd = self._run_eval(
            ["-c", "-m", "ndcg_cut.10", str(self.qrels), str(self.run)]
        )
        self.assertEqual(
            cmd,
            [
                "java",
                "-jar",
                "trec_eval.jar",
                "-c",
                "-m",
                "ndcg_cut.10",
                str(self.qrels),
                str(self.run),
            ],
        )

    def test_eval_keeps_metric_flag_without_leading_c(self):
        cmd = self._run_eval(["-m", "recall.20", str(self.qrels), str(self.run)])
        self.assertEqual(
            cmd,
            [
                "java",
                "-jar",
                "trec_eval.jar",
                "-m",
                "recall.20",
                str(self.qrels),
                str(self.run),
            ],
        )

    def test_from_trec_runfile_forwards_default_eval_args(self):
        cmd = self._captured_cmd(
            lambda: EvalFunction.from_trec_runfile(str(self.run), str(self.qrels))
        )
        self.assertEqual(cmd[3:6], ["-c", "-m", "ndcg_cut.10"])


if __name__ == "__main__":
    unittest.main()

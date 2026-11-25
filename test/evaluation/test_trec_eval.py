import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

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
                    "candidates": [{"doc": {"text": "Doc3"}, "docid": "D3", "score": 0.85}],
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
        self.qrels.write_text("q1 0 D1 1\nq2 0 D2 1\n", encoding="utf-8")
        self.run.write_text("q1 Q0 D1 1 0.9 rank_llm\n", encoding="utf-8")

    def _captured_args(self, args):
        original = list(args)
        with patch("src.rank_llm.evaluation.trec_eval._require_trec_eval") as loader:
            EvalFunction.eval(args)
        self.assertEqual(args, original)
        return loader.return_value.call_args.args[0]

    def test_eval_keeps_first_option(self):
        args = ["-c", "-m", "ndcg_cut.10", str(self.qrels), str(self.run)]
        self.assertEqual(self._captured_args(args), ["trec_eval", *args])

    def test_eval_keeps_metric_flag_without_leading_c(self):
        args = ["-m", "recall.20", str(self.qrels), str(self.run)]
        self.assertEqual(self._captured_args(args), ["trec_eval", *args])

    def test_delegates_special_options(self):
        args = ["-remove-unjudged", "-m", "judged.5,10", str(self.qrels), str(self.run)]
        self.assertEqual(self._captured_args(args), ["trec_eval", *args])

    def test_delegates_trailing_remove_unjudged(self):
        args = [
            "-c",
            "-m",
            "ndcg_cut.10",
            str(self.qrels),
            str(self.run),
            "-remove-unjudged",
        ]
        self.assertEqual(self._captured_args(args), ["trec_eval", *args])

    def test_from_trec_runfile_forwards_defaults_and_original_qrels(self):
        with patch("src.rank_llm.evaluation.trec_eval._require_trec_eval") as loader:
            EvalFunction.from_trec_runfile(str(self.run), str(self.qrels))
        args = loader.return_value.call_args.args[0]
        self.assertEqual(args[:4], ["trec_eval", "-c", "-m", "ndcg_cut.10"])
        self.assertEqual(args[-2], str(self.qrels))
        self.assertTrue(self.qrels.exists())

    def test_preserves_multiple_metrics_and_prints_once(self):
        output = "ndcg_cut_10\tall\t0.5000\nrecall_20\tall\t0.7500"

        def evaluate(args, **kwargs):
            print(output)
            args.pop()
            return 0.75

        args = ["-m", "ndcg_cut.10", "-m", "recall.20", str(self.qrels), str(self.run)]
        original = list(args)
        captured = StringIO()
        with (
            patch(
                "src.rank_llm.evaluation.trec_eval._require_trec_eval",
                return_value=evaluate,
            ),
            redirect_stdout(captured),
        ):
            result = EvalFunction.eval(args)
        self.assertEqual(result, output)
        self.assertEqual(captured.getvalue(), output + "\n")
        self.assertEqual(args, original)

    def test_per_query_only_output_uses_pyserini_parser(self):
        from pyserini.eval.trec_eval import trec_eval

        output = "ndcg_cut_10\tq1\t0.5000"
        args = ["-q", "-n", "-m", "ndcg_cut.10", str(self.qrels), str(self.run)]
        captured = StringIO()
        with (
            patch(
                "src.rank_llm.evaluation.trec_eval._require_trec_eval",
                return_value=trec_eval,
            ),
            patch("pyserini.eval.trec_eval.subprocess.Popen") as process,
            redirect_stdout(captured),
        ):
            process.return_value.communicate.return_value = (output.encode(), b"")
            result = EvalFunction.eval(args)
        self.assertEqual(result, output)
        self.assertEqual(captured.getvalue(), output + "\n")
        self.assertEqual(process.call_args.args[0][-len(args) :], args)

    def test_failure_propagates_and_preserves_original_qrels(self):
        with patch("src.rank_llm.evaluation.trec_eval._require_trec_eval") as loader:
            loader.return_value.side_effect = RuntimeError("evaluation failed")
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                EvalFunction.from_trec_runfile(str(self.run), str(self.qrels))
        args = loader.return_value.call_args.args[0]
        self.assertEqual(args[-2], str(self.qrels))
        self.assertTrue(self.qrels.exists())

    def test_failure_cleans_temporary_run(self):
        with patch.object(
            EvalFunction, "eval", side_effect=RuntimeError("failed")
        ) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                EvalFunction.from_results([], str(self.qrels))
        self.assertFalse(Path(evaluate.call_args.args[0][-1]).exists())


if __name__ == "__main__":
    unittest.main()

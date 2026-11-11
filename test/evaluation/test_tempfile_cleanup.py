"""Regression tests for temporary TREC files created during evaluation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.rank_llm.evaluation.trec_eval import EvalFunction


class TestEvalTempFileCleanup(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.qrels = self.directory / "qrels.txt"
        self.run = self.directory / "run.txt"
        self.qrels.write_text("q1 0 D1 1\n", encoding="utf-8")
        self.run.write_text("q1 Q0 D1 1 0.9 rank_llm\n", encoding="utf-8")

    def test_from_results_cleans_temporary_run_on_success(self):
        data = [
            SimpleNamespace(
                query=SimpleNamespace(qid="q1"),
                candidates=[SimpleNamespace(docid="D1", score=0.9)],
            )
        ]
        with patch.object(EvalFunction, "eval", return_value="OK") as mock_eval:
            self.assertEqual(EvalFunction.from_results(data, str(self.qrels)), "OK")
        temporary_run = Path(mock_eval.call_args.args[0][-1])
        self.assertFalse(temporary_run.exists())
        self.assertTrue(self.run.exists())

    def test_from_results_cleans_temporary_run_on_failure(self):
        data = [
            SimpleNamespace(
                query=SimpleNamespace(qid="q1"),
                candidates=[SimpleNamespace(docid="D1", score=0.9)],
            )
        ]
        with patch.object(EvalFunction, "eval", side_effect=RuntimeError("failed")) as mock_eval:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                EvalFunction.from_results(data, str(self.qrels))
        temporary_run = Path(mock_eval.call_args.args[0][-1])
        self.assertFalse(temporary_run.exists())

    def _evaluate_with_truncated_qrels(self, fail=False):
        truncated = self.directory / "truncated_qrels.txt"
        truncated.write_text("q1 0 D1 1\n", encoding="utf-8")
        args = ["-c", "-m", "ndcg_cut.10", str(self.qrels), str(self.run)]
        with (
            patch(
                "src.rank_llm.evaluation.trec_eval.download_evaluation_script",
                return_value="trec_eval.jar",
            ),
            patch(
                "src.rank_llm.evaluation.trec_eval._require_pandas",
                return_value=MagicMock(),
            ),
            patch.object(EvalFunction, "trunc", return_value=str(truncated)),
            patch("src.rank_llm.evaluation.trec_eval.subprocess.Popen") as popen,
        ):
            if fail:
                popen.side_effect = OSError("evaluation unavailable")
                with self.assertRaisesRegex(OSError, "evaluation unavailable"):
                    EvalFunction.eval(args)
            else:
                popen.return_value.communicate.return_value = (
                    b"ndcg_cut_10 all 0.9",
                    b"",
                )
                self.assertEqual(EvalFunction.eval(args), "ndcg_cut_10 all 0.9")
        self.assertFalse(truncated.exists())
        self.assertTrue(self.qrels.exists())
        self.assertTrue(self.run.exists())

    def test_eval_cleans_truncated_qrels_on_success(self):
        self._evaluate_with_truncated_qrels()

    def test_eval_cleans_truncated_qrels_on_subprocess_error(self):
        self._evaluate_with_truncated_qrels(fail=True)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the lightweight RankLLM evaluation-comparison helper."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


COMPARE_PATH = (
    Path(__file__).resolve().parents[1]
    / ".claude/skills/rank-llm-eval/scripts/compare.py"
)
SPEC = importlib.util.spec_from_file_location("rankllm_eval_compare", COMPARE_PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class TestEvaluationComparison(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def make_file(self, name, records):
        p = self.path / name
        p.write_text("\n".join(json.dumps(x) for x in records) + "\n", encoding="utf-8")
        return str(p)

    def test_old_and_new_formats_produce_same_metric_rows(self):
        old = self.make_file("old.jsonl", [{
            "file": "dl19/run.txt",
            "result": [
                "ndcg_cut_10 \tall\t0.7043",
                "map_cut_100   \tall\t0.4120",
            ],
        }])
        new = self.make_file("new.jsonl", [{
            "file": "dl19/run.txt",
            "result": [
                {"metric": "ndcg_cut_10", "value": "0.7043"},
                {"metric": "map_cut_100", "value": "0.4120"},
            ],
        }])
        self.assertEqual(compare.load_metrics(old), compare.load_metrics(new))
        self.assertEqual(len(compare.compare_metrics(compare.load_metrics(old), compare.load_metrics(new))), 2)

    def test_regression_gate_and_markdown(self):
        baseline = self.make_file("baseline.jsonl", [{
            "file": "dl19/run.txt", "result": [{"metric": "ndcg_cut_10", "value": 0.75}]
        }])
        candidate = self.make_file("candidate.jsonl", [{
            "file": "dl19/run.txt", "result": ["ndcg_cut_10 all 0.73"]
        }])
        from contextlib import redirect_stdout
        from io import StringIO
        output = StringIO()
        with redirect_stdout(output):
            result = compare.main([
                "--run-a", baseline, "--run-b", candidate,
                "--metrics", "ndcg_cut_10", "--max-drop", "0.01", "--format", "markdown",
            ])
        self.assertEqual(result, 2)
        self.assertIn("| FAIL |", output.getvalue())
        self.assertIn("-0.0200", output.getvalue())
        with redirect_stdout(StringIO()):
            self.assertEqual(compare.main([
                "--run-a", baseline, "--run-b", candidate, "--max-drop", "0.03",
            ]), 0)

    def test_missing_metric_does_not_silently_succeed(self):
        a = self.make_file("a.jsonl", [{"file": "run.txt", "result": ["ndcg_cut_10 all 0.7"]}])
        b = self.make_file("b.jsonl", [{"file": "run.txt", "result": ["map_cut_100 all 0.2"]}])
        with self.assertRaisesRegex(ValueError, "No common metrics"):
            compare.compare_metrics(compare.load_metrics(a), compare.load_metrics(b))

    def test_require_complete_detects_file_mismatch(self):
        a = {"dl19": {"ndcg_cut_10": 0.7}, "dl20": {"ndcg_cut_10": 0.6}}
        b = {"dl19": {"ndcg_cut_10": 0.7}}
        with self.assertRaisesRegex(ValueError, "Run files differ"):
            compare.compare_metrics(a, b, require_complete=True)

    def test_rejects_duplicate_file_and_non_finite_metric(self):
        dup = self.make_file("dup.jsonl", [
            {"file": "run.txt", "result": ["ndcg_cut_10 all 0.7"]},
            {"file": "run.txt", "result": ["ndcg_cut_10 all 0.8"]},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate file"):
            compare.load_metrics(dup)
        bad = self.make_file("nan.jsonl", [
            {"file": "run.txt", "result": ["ndcg_cut_10 all NaN"]},
        ])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare.load_metrics(bad)

    def test_rejects_partial_or_invalid_records(self):
        p = self.make_file("bad.jsonl", [{"file": "run.txt", "result": ["nonsense"]}])
        with self.assertRaisesRegex(ValueError, "expected 'metric all value'"):
            compare.load_metrics(p)


if __name__ == "__main__":
    unittest.main()

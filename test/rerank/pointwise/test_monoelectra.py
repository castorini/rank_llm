import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rank_llm.rerank.pointwise.monoelectra import MonoELECTRA
from rank_llm.rerank.reranker import TEMPLATES, Reranker


@patch("rank_llm.rerank.pointwise.monoelectra.AutoModelForSequenceClassification")
@patch("rank_llm.rerank.pointwise.monoelectra.AutoTokenizer")
class TestMonoELECTRAFactory(unittest.TestCase):
    def test_default_template_loads_outside_repo_root(self, *_):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            coordinator = Reranker.create_model_coordinator(
                "monoelectra", None, False, device="cpu"
            )

        self.assertIsInstance(coordinator, MonoELECTRA)

    def test_accepts_path_prompt_template_path(self, *_):
        # run_rank_llm.py passes the template as a pathlib.Path.
        coordinator = Reranker.create_model_coordinator(
            "monoelectra",
            None,
            False,
            device="cpu",
            prompt_template_path=Path(TEMPLATES / "monoelectra_template.yaml"),
        )

        self.assertIsInstance(coordinator, MonoELECTRA)


if __name__ == "__main__":
    unittest.main()

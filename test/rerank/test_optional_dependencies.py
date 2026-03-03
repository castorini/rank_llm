import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from rank_llm import doctor
from rank_llm.rerank import Reranker
from rank_llm.rerank import reranker as reranker_module


class TestOptionalDependencies(unittest.TestCase):
    def test_reranker_import_without_optional_backends(self):
        self.assertIsNotNone(Reranker)

    def test_openai_dependency_error_message(self):
        with patch.object(
            reranker_module,
            "_load_safe_openai",
            side_effect=ImportError("missing openai"),
        ):
            with self.assertRaises(ImportError) as ctx:
                Reranker.create_model_coordinator(
                    model_path="gpt-4o-mini",
                    default_model_coordinator=None,
                    interactive=False,
                )
        self.assertIn("missing openai", str(ctx.exception))

    def test_doctor_output_contains_core_sections(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            doctor.main()
        output = buffer.getvalue()
        self.assertIn("RankLLM doctor", output)
        self.assertIn("Optional backend checks:", output)
        self.assertIn("Environment variables:", output)


if __name__ == "__main__":
    unittest.main()

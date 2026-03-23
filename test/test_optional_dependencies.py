import unittest
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import rank_llm.retrieve as retrieve
import rank_llm.rerank.listwise as listwise
from rank_llm._optional import install_hint


class TestOptionalDependencies(unittest.TestCase):
    def test_retrieve_lazy_imports_pyserini_symbols(self):
        module = SimpleNamespace(PyseriniRetriever="fake-retriever")

        with patch("rank_llm.retrieve.import_module", return_value=module) as importer:
            value = retrieve.__getattr__("PyseriniRetriever")

        self.assertEqual(value, "fake-retriever")
        importer.assert_called_once_with("rank_llm.retrieve.pyserini_retriever")

    def test_listwise_lazy_imports_safe_openai(self):
        module = SimpleNamespace(SafeOpenai="fake-safe-openai")

        with patch("rank_llm.rerank.listwise.import_module", return_value=module) as importer:
            value = listwise.__getattr__("SafeOpenai")

        self.assertEqual(value, "fake-safe-openai")
        importer.assert_called_once_with("rank_llm.rerank.listwise.rank_gpt")

    def test_safe_openai_reports_openai_extra(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            rank_gpt = import_module("rank_llm.rerank.listwise.rank_gpt")

        with patch.object(rank_gpt, "openai", None), patch.object(rank_gpt, "tiktoken", None):
            with self.assertRaises(ImportError) as exc:
                rank_gpt.SafeOpenai(model="gpt-4o-mini", context_size=4096, keys="test")

        self.assertIn(install_hint("openai"), str(exc.exception))

    def test_safe_genai_reports_genai_extra(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            rank_gemini = import_module("rank_llm.rerank.listwise.rank_gemini")

        with patch.object(rank_gemini.ListwiseRankLLM, "__init__", return_value=None), patch.object(rank_gemini, "genai", None):
            with self.assertRaises(ImportError) as exc:
                rank_gemini.SafeGenai(
                    model="gemini-2.0-flash-001",
                    context_size=4096,
                    prompt_template_path="unused.yaml",
                    keys="test",
                )

        self.assertIn(install_hint("genai"), str(exc.exception))


if __name__ == "__main__":
    unittest.main()

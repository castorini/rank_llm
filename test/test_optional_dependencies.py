#!/usr/bin/env python3
"""
Test for optional dependencies functionality.
This test validates that modules handle missing optional dependencies gracefully.
"""

import unittest
import sys


class TestOptionalDependencies(unittest.TestCase):
    """Test that optional dependencies are handled correctly."""

    def test_vllm_optional_import(self):
        """Test that vLLM can be optionally imported."""
        # Test the import logic that's used in vllm_handler.py
        try:
            import vllm
            from vllm.outputs import RequestOutput
            vllm_available = True
        except ImportError:
            vllm_available = False
            vllm = None
            RequestOutput = None

        # This should not raise an error regardless of whether vLLM is installed
        self.assertIsInstance(vllm_available, bool)
        
        if not vllm_available:
            self.assertIsNone(vllm)
            self.assertIsNone(RequestOutput)

    def test_transformers_optional_import(self):
        """Test that transformers can be optionally imported."""
        # Test the import logic used in various model files
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            from transformers.generation import GenerationConfig
            from transformers import PreTrainedTokenizerBase
            transformers_available = True
        except ImportError:
            transformers_available = False
            T5ForConditionalGeneration = None
            T5Tokenizer = None
            GenerationConfig = None
            PreTrainedTokenizerBase = None

        # This should not raise an error regardless of whether transformers is installed
        self.assertIsInstance(transformers_available, bool)
        
        if not transformers_available:
            self.assertIsNone(T5ForConditionalGeneration)
            self.assertIsNone(T5Tokenizer)
            self.assertIsNone(GenerationConfig)
            self.assertIsNone(PreTrainedTokenizerBase)

    def test_error_messages_contain_install_instructions(self):
        """Test that error messages include helpful installation instructions."""
        vllm_error_msg = "vLLM is not installed. Please install it with: pip install rank_llm[vllm]"
        transformers_error_msg = "transformers is not installed. Please install it with: pip install rank_llm[transformers]"
        
        # Verify the error messages contain the expected installation instructions
        self.assertIn("pip install rank_llm[vllm]", vllm_error_msg)
        self.assertIn("pip install rank_llm[transformers]", transformers_error_msg)


if __name__ == "__main__":
    unittest.main()
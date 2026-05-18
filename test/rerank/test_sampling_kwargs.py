"""Unit tests for OpenAI-vs-extra_body splitting of sampling kwargs."""

from __future__ import annotations

import unittest

from rank_llm.rerank.sampling_kwargs import (
    sanitize_sampling_kwargs,
    split_openai_chat_sampling,
)


class TestSamplingKwargs(unittest.TestCase):
    def test_sanitize_drops_owned_keys(self):
        inp = {"temperature": 0.8, "max_tokens": 99, "repetition_penalty": 1.1}
        out = sanitize_sampling_kwargs(inp)
        self.assertEqual(out, {"temperature": 0.8, "repetition_penalty": 1.1})

    def test_split_openai_vs_extra_body(self):
        extras = sanitize_sampling_kwargs(
            {
                "temperature": 0.7,
                "top_p": 0.92,
                "top_k": 40,
                "repetition_penalty": 1.15,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                "stop": ["</think>"],
            }
        )
        direct, eb = split_openai_chat_sampling(extras)
        self.assertEqual(direct["temperature"], 0.7)
        self.assertEqual(eb["top_k"], 40)
        self.assertEqual(eb["repetition_penalty"], 1.15)


if __name__ == "__main__":
    unittest.main()

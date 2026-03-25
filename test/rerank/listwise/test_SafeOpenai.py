import re
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rank_llm.rerank.listwise import SafeOpenai
from rank_llm.rerank.listwise import rank_gpt as _rank_gpt_module
from rank_llm.rerank.rankllm import PromptMode

# model, context_size, prompt_template_path, num_few_shot_examples, keys, key_start_id
valid_inputs = [
    (
        "gpt-3.5-turbo",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_gpt_template.yaml",
        0,
        "OPEN_AI_API_KEY",
        None,
    ),
    (
        "gpt-3.5-turbo",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_lrl_template.yaml",
        0,
        "OPEN_AI_API_KEY",
        3,
    ),
    (
        "gpt-4",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_gpt_template.yaml",
        0,
        "OPEN_AI_API_KEY",
        None,
    ),
    (
        "gpt-4",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_lrl_template.yaml",
        0,
        "OPEN_AI_API_KEY",
        3,
    ),
]

failure_inputs = [
    ("gpt-3.5-turbo", 4096, PromptMode.RANK_GPT, 0, None),  # missing key
    ("gpt-3.5-turbo", 4096, PromptMode.LRL, 0, None),  # missing key
    (
        "gpt-3.5-turbo",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        "OPEN_AI_API_KEY",
    ),  # unpecified prompt mode
    ("gpt-4", 4096, PromptMode.RANK_GPT, 0, None),  # missing key
    ("gpt-4", 4096, PromptMode.LRL, 0, None),  # missing key
    (
        "gpt-4",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        "OPEN_AI_API_KEY",
    ),  # unpecified prompt mode
]


class TestSafeOpenai(unittest.TestCase):
    def setUp(self):
        # Provide lightweight stubs so SafeOpenai can be constructed without
        # the real openai/tiktoken packages installed.
        self._orig_openai = _rank_gpt_module.openai
        self._orig_tiktoken = _rank_gpt_module.tiktoken
        if _rank_gpt_module.openai is None:
            _rank_gpt_module.openai = SimpleNamespace(
                proxy=None,
                api_key=None,
                base_url=None,
                api_version=None,
                api_type=None,
                api_base=None,
                responses=MagicMock(),
            )
        if _rank_gpt_module.tiktoken is None:
            mock_enc = MagicMock()
            mock_enc.encode.side_effect = lambda t: re.findall(
                r"\d+|[A-Za-z]+|[^\sA-Za-z\d_]", t
            )
            mock_tiktoken = MagicMock()
            mock_tiktoken.get_encoding.return_value = mock_enc
            _rank_gpt_module.tiktoken = mock_tiktoken

    def tearDown(self):
        _rank_gpt_module.openai = self._orig_openai
        _rank_gpt_module.tiktoken = self._orig_tiktoken

    def test_valid_inputs(self):
        for (
            model,
            context_size,
            prompt_template_path,
            num_few_shot_examples,
            keys,
            key_start_id,
        ) in valid_inputs:
            obj = SafeOpenai(
                model=model,
                context_size=context_size,
                prompt_template_path=prompt_template_path,
                num_few_shot_examples=num_few_shot_examples,
                keys=keys,
            )
            self.assertEqual(obj._model, model)
            self.assertEqual(obj._context_size, context_size)
            self.assertEqual(obj._num_few_shot_examples, num_few_shot_examples)
            self.assertEqual(obj._keys[0], keys)
            if key_start_id is not None:
                self.assertEqual(obj._cur_key_id, key_start_id % 1)
            else:
                self.assertEqual(obj._cur_key_id, 0)

    def test_failure_inputs(self):
        for (
            model,
            context_size,
            prompt_mode,
            num_few_shot_examples,
            keys,
        ) in failure_inputs:
            with self.assertRaises(ValueError):
                SafeOpenai(
                    model=model,
                    context_size=context_size,
                    prompt_mode=prompt_mode,
                    num_few_shot_examples=num_few_shot_examples,
                    keys=keys,
                )

    @patch("rank_llm.rerank.listwise.rank_gpt.SafeOpenai._call_completion")
    def test_run_llm(self, mock_call_completion):
        mock_call_completion.return_value = "mock_response"
        model_coordinator = SafeOpenai(
            model="gpt-3.5",
            context_size=4096,
            prompt_template_path="src/rank_llm/rerank/prompt_templates/rank_gpt_template.yaml",
            num_few_shot_examples=0,
            keys="OPEN_AI_API_KEY",
        )

        output, size = model_coordinator.run_llm("how are you?")
        self.assertEqual(output, "mock_response")
        self.assertEqual(size, 2)

    def test_num_output_tokens(self):
        model_coordinator = SafeOpenai(
            model="gpt-3.5",
            context_size=4096,
            prompt_template_path="src/rank_llm/rerank/prompt_templates/rank_gpt_template.yaml",
            num_few_shot_examples=0,
            keys="OPEN_AI_API_KEY",
        )

        output = model_coordinator.num_output_tokens(current_window_size=5)
        self.assertEqual(output, 18)

        output = model_coordinator.num_output_tokens(current_window_size=1)
        self.assertEqual(output, 2)


if __name__ == "__main__":
    unittest.main()

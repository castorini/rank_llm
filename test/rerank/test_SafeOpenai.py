import unittest
from unittest.mock import patch

from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.rankllm import PromptMode

# model, context_size, prompt_mode, num_few_shot_examples, keys, key_start_id
valid_inputs = [
    ("gpt-3.5-turbo", 4096, PromptMode.RANK_GPT, 0, "OPEN_AI_API_KEY", None),
    ("gpt-3.5-turbo", 4096, PromptMode.LRL, 0, "OPEN_AI_API_KEY", 3),
    ("gpt-4", 4096, PromptMode.RANK_GPT, 0, "OPEN_AI_API_KEY", None),
    ("gpt-4", 4096, PromptMode.LRL, 0, "OPEN_AI_API_KEY", 3),
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
    def test_valid_inputs(self):
        for (
            model,
            context_size,
            prompt_mode,
            num_few_shot_examples,
            keys,
            key_start_id,
        ) in valid_inputs:
            obj = SafeOpenai(
                model=model,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                keys=keys,
            )
            self.assertEqual(obj._model, model)
            self.assertEqual(obj._context_size, context_size)
            self.assertEqual(obj._prompt_mode, prompt_mode)
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
            with self.assertRaises(BaseException):
                obj = SafeOpenai(
                    model=model,
                    context_size=context_size,
                    prompt_mode=prompt_mode,
                    num_few_shot_examples=num_few_shot_examples,
                    keys=keys,
                )

    @patch("rank_llm.rerank.rank_gpt.SafeOpenai._call_completion")
    def test_run_llm(self, mock_call_completion):
        mock_call_completion.return_value = "mock_response"
        agent = SafeOpenai(
            model="gpt-3.5",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            keys="OPEN_AI_API_KEY",
        )

        output, size = agent.run_llm("how are you?")
        self.assertEqual(output, "mock_response")
        self.assertEqual(size, 2)

    def test_num_output_tokens(self):
        agent = SafeOpenai(
            model="gpt-3.5",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            keys="OPEN_AI_API_KEY",
        )

        output = agent.num_output_tokens(current_window_size=5)
        self.assertEqual(output, 18)

        output = agent.num_output_tokens(current_window_size=1)
        self.assertEqual(output, 2)


if __name__ == "__main__":
    unittest.main()

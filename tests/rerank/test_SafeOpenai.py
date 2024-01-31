from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.rankllm import RankLLM, PromptMode

import unittest
from unittest.mock import patch, MagicMock
from rank_llm.result import Result

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


def run_valid_input_tests(inputs):
    for (
        model,
        context_size,
        prompt_mode,
        num_few_shot_examples,
        keys,
        key_start_id,
    ) in inputs:
        obj = SafeOpenai(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            keys=keys,
        )
        assert obj._model == model
        assert obj._context_size == context_size
        assert obj._prompt_mode == prompt_mode
        assert obj._num_few_shot_examples == num_few_shot_examples
        assert obj._keys[0] == keys
        if key_start_id is not None:
            assert obj._cur_key_id == key_start_id % 1
        else:
            assert obj._cur_key_id == 0

    print("\033[92mValid inputs tests passed\033[0m")


def run_failure_input_tests(inputs):
    count = 0
    for model, context_size, prompt_mode, num_few_shot_examples, keys in inputs:
        try:
            obj = SafeOpenai(
                model=model,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                keys=keys,
            )
        except:
            print("Exception raised correctly")
            count += 1

    if count == len(inputs):
        print(f"\033[92m{count}/{len(inputs)} exceptions raised correctly\033[0m")
    else:
        print(f"\033[91m{count}/{len(inputs)} exceptions raised correctly\033[0m")


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

    def test_run_llm(self):
        agent = SafeOpenai(
            model="gpt-3.5",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            keys="OPEN_AI_API_KEY",
        )

        # output, size = agent.run_llm("how are you?")
        # print(output, size)

        # obj._llm = MagicMock()
        # obj._tokenizer = MagicMock()
        # obj._llm.generate.return_value = (
        #     [MagicMock()],
        #     [MagicMock()],
        # )
        # obj._tokenizer.decode.return_value = "test"
        # obj._tokenizer.decode.side_effect

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

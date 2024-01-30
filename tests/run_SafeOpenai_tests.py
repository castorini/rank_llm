from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.rankllm import RankLLM, PromptMode


# --estimation_mode=create_prpts --model_name=gpt-3.5-turbo --prompt_mode=rank_GPT

# model, context_size, prompt_mode, num_few_shot_examples, keys, key_start_id
valid_inputs = [
    ("gpt-3.5-turbo", 4096, PromptMode.RANK_GPT, 0, "OPEN_AI_API_KEY", None),
    ("gpt-3.5-turbo", 4096, PromptMode.LRL, 0, "OPEN_AI_API_KEY", 3),
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

    print("Valid inputs tests passed")


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

    print(f"{count}/{len(inputs)} exceptions raised correctly")


if __name__ == "__main__":
    run_valid_input_tests(valid_inputs)
    run_failure_input_tests(failure_inputs)

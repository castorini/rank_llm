from rank_llm.rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.rankllm import PromptMode

# model, context_size, prompt_mode, num_few_shot_examples, variable_passages, window_size, system_message
valid_inputs = [
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.RANK_GPT,
        0,
        True,
        10,
        "Default Message",
    ),
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.RANK_GPT,
        0,
        False,
        10,
        "Default Message",
    ),
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.RANK_GPT,
        0,
        True,
        30,
        "Default Message",
    ),
    ("castorini/rank_zephyr_7b_v1_full", 4096, PromptMode.RANK_GPT, 0, True, 10, ""),
    ("castorini/rank_vicuna_7b_v1", 4096, PromptMode.RANK_GPT, 0, True, 10, ""),
    ("castorini/rank_vicuna_7b_v1_noda", 4096, PromptMode.RANK_GPT, 0, True, 10, ""),
    ("castorini/rank_vicuna_7b_v1_fp16", 4096, PromptMode.RANK_GPT, 0, True, 10, ""),
    ("castorini/rank_vicuna_7b_v1_noda_fp16", 4096, PromptMode.RANK_GPT, 0, True, 10, ""),
]

failure_inputs = [
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1",
        4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_noda",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    ( "castorini/rank_vicuna_7b_v1_noda", 4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_fp16",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_fp16",
        4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_noda_fp16",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_noda_fp16",
        4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
]


def run_valid_input_tests(inputs):
    for (
        model,
        context_size,
        prompt_mode,
        num_few_shot_examples,
        variable_passages,
        window_size,
        system_message,
    ) in inputs:
        obj = RankListwiseOSLLM(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            variable_passages=variable_passages,
            window_size=window_size,
            system_message=system_message,
        )
        assert obj._model == model
        assert obj._context_size == context_size
        assert obj._prompt_mode == prompt_mode
        assert obj._num_few_shot_examples == num_few_shot_examples
        assert obj._variable_passages == variable_passages
        assert obj._window_size == window_size
        assert obj._system_message == system_message

    print("\033[92mValid inputs tests passed\033[0m")


def run_failure_input_tests(inputs):
    count = 0
    for (
        model,
        context_size,
        prompt_mode,
        num_few_shot_examples,
        variable_passages,
        window_size,
        system_message,
    ) in inputs:
        try:
            obj = RankListwiseOSLLM(
                model=model,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                variable_passages=variable_passages,
                window_size=window_size,
                system_message=system_message,
            )
        except:
            print("Exception raised correctly")
            count += 1
    
    if count == len(inputs):
        print(f"\033[92m{count}/{len(inputs)} exceptions raised correctly\033[0m")
    else:
        print(f"\033[91m{count}/{len(inputs)} exceptions raised correctly\033[0m")


if __name__ == "__main__":
    run_valid_input_tests(valid_inputs)
    run_failure_input_tests(failure_inputs)

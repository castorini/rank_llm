import unittest
from unittest.mock import MagicMock, patch

from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank import PromptMode
from rank_llm.rerank.listwise.multiturn_listwise_inference_handler import (
    MultiTurnListwiseInferenceHandler,
)
from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.listwise.singleturn_listwise_inference_handler import (
    SingleTurnListwiseInferenceHandler,
)

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
    (
        "castorini/rank_vicuna_7b_v1_noda_fp16",
        4096,
        PromptMode.RANK_GPT,
        0,
        True,
        10,
        "",
    ),
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
    (
        "castorini/rank_vicuna_7b_v1_noda",
        4096,
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


r = from_dict(
    data_class=Result,
    data={
        "query": {"text": "Sample Query", "qid": "q1"},
        "candidates": [
            {
                "doc": {
                    "contents": "Title: Sample Title Content: Sample Text",
                },
                "docid": "d1",
                "score": 0.5,
            },
            {
                "doc": {
                    "contents": "Title: Sample Title Content: Sample Text",
                },
                "docid": "d2",
                "score": 0.4,
            },
            {
                "doc": {
                    "contents": "Title: Sample Title Content: Sample Text",
                },
                "docid": "d3",
                "score": 0.4,
            },
            {
                "doc": {
                    "contents": "Title: Sample Title Content: Sample Text",
                },
                "docid": "d4",
                "score": 0.3,
            },
        ],
    },
)


class TestRankListwiseOSLLM(unittest.TestCase):
    def setUp(self):
        # Patch cuda availability check
        self.patcher_cuda = patch("torch.cuda.is_available", return_value=True)
        self.mock_cuda = self.patcher_cuda.start()

        # Mock Tokenizer with apply_chat_template method
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template.side_effect = (
            lambda messages, **kwargs: str(messages)
        )

        # Mock vllm.LLM
        self.patcher_vllm = patch("vllm.LLM", autospec=True)
        self.mock_vllm_class = self.patcher_vllm.start()
        self.mock_vllm_instance = self.mock_vllm_class.return_value
        self.mock_vllm_instance.get_tokenizer.return_value = self.mock_tokenizer

        # Mock generate method
        self.mock_vllm_instance.generate.return_value = ["Mock response"]

    def tearDown(self):
        self.patcher_cuda.stop()
        self.patcher_vllm.stop()

    def test_valid_inputs(self):
        for (
            model,
            context_size,
            prompt_mode,
            num_few_shot_examples,
            variable_passages,
            window_size,
            system_message,
        ) in valid_inputs:
            model_coordinator = RankListwiseOSLLM(
                model=model,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                variable_passages=variable_passages,
                window_size=window_size,
                system_message=system_message,
            )
            self.assertEqual(model_coordinator._model, model)
            self.assertEqual(model_coordinator._context_size, context_size)
            self.assertEqual(model_coordinator._prompt_mode, prompt_mode)
            self.assertEqual(
                model_coordinator._num_few_shot_examples, num_few_shot_examples
            )
            self.assertEqual(model_coordinator._variable_passages, variable_passages)
            self.assertEqual(model_coordinator._window_size, window_size)
            self.assertEqual(model_coordinator._system_message, system_message)

    def test_failure_inputs(self):
        for (
            model,
            context_size,
            prompt_mode,
            num_few_shot_examples,
            variable_passages,
            window_size,
            system_message,
        ) in failure_inputs:
            with self.assertRaises(ValueError):
                model_coordinator = RankListwiseOSLLM(
                    model=model,
                    context_size=context_size,
                    prompt_mode=prompt_mode,
                    num_few_shot_examples=num_few_shot_examples,
                    variable_passages=variable_passages,
                    window_size=window_size,
                    system_message=system_message,
                )

    @patch(
        "rank_llm.rerank.listwise.rank_listwise_os_llm.RankListwiseOSLLM.num_output_tokens"
    )
    def test_num_output_tokens(self, mock_num_output_tokens):
        # Creating PyseriniRetriever instance
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            name="rank_zephyr",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            variable_passages=True,
            window_size=10,
            system_message="",
        )

        mock_num_output_tokens.return_value = 40
        output = model_coordinator.num_output_tokens()
        self.assertEqual(output, 40)

        # print(output)
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            name="rank_zephyr",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            variable_passages=True,
            window_size=5,
            system_message="",
        )

        mock_num_output_tokens.return_value = 19
        output = model_coordinator.num_output_tokens()
        self.assertEqual(output, 19)

    @patch(
        "rank_llm.rerank.listwise.rank_listwise_os_llm.RankListwiseOSLLM.run_llm_batched"
    )
    def test_run_llm_batched(self, mock_run_llm_batched):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            name="rank_zephyr",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            variable_passages=True,
            window_size=5,
            system_message="",
        )

        mock_run_llm_batched.return_value = ("> [1] > [2] > [3] > [4] > [5", 19)
        output, size = model_coordinator.run_llm_batched(
            "How are you doing ? What is your name? What is your age? What is your favorite color?"
        )
        expected_output = "> [1] > [2] > [3] > [4] > [5"
        self.assertEqual(output, expected_output)
        self.assertEqual(size, len([char for char in output if char != " "]))

    def test_create_prompt(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            name="rank_zephyr",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            variable_passages=True,
            window_size=5,
            system_message="",
            device="cpu",
        )

        import re

        def get_first_int(s):
            match = re.search(r"\d+", s)
            return int(match.group()) if match else None

        start_end_pairs = [(1, 3), (2, 4), (3, 5), (5, 6)]
        for start, end in start_end_pairs:
            prompt, length = model_coordinator.create_prompt(r, start, end)
            expected_output = min(end, len(r.candidates)) - max(0, start)
            self.assertEqual(get_first_int(prompt), max(expected_output, 0))

    @patch(
        "rank_llm.rerank.listwise.rank_listwise_os_llm.RankListwiseOSLLM.get_num_tokens"
    )
    def test_get_num_tokens(self, mock_get_num_tokens):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            name="rank_zephyr",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            variable_passages=True,
            window_size=5,
            system_message="",
            device="cpu",
        )

        mock_get_num_tokens.return_value = 22
        output = model_coordinator.get_num_tokens(
            "How are you doing? What is your name? What is your age? What is your favorite color?"
        )
        self.assertEqual(output, 22)


VALID_SINGLETURN_TEMPLATE = {
    "method": "singleturn_listwise",
    "system_message": "You are a helpful assistant that ranks documents.",
    "prefix": "Sample prefix: Rank these {num} passages for query: {query}",
    "suffix": "Sample suffix: Rank the provided {num} passages based on query: {query}",
    "body": "[{rank}] {candidate}\n",
}
VALID_MULTITURN_TEMPLATE_1 = {
    "method": "multiturn_listwise",
    "system_message": "You are a helpful assistant than ranks documents.",
    "prefix_user": "Sample prefix: Rank these {num} passages for query: {query}",
    "prefix_assistant": "Okay, please provide the passages.",
    "body_user": "[{rank}] {candidate}",
    "body_assistant": "Received passage [{rank}].",
    "suffix_user": "Sample suffix: Rank the provided {num} passages based on query: {query}",
}
VALID_MULTITURN_TEMPLATE_2 = {
    "method": "multiturn_listwise",
    "system_message": "You are a helpful assistant than ranks documents.",
    "prefix_user": "Sample prefix: Rank these {num} passages for query: {query}",
    "prefix_assistant": "Okay, please provide the passages.",
    "body_user": "[{rank}] {candidate}\n",
    "suffix_user": "Sample suffix: Rank the provided {num} passages based on query: {query}",
}
VALID_LRL_TEMPLATE = {
    "method": "singleturn_listwise",
    "prefix": "Sample prefix: {query}",
    "body": "[{rank}] {candidate}\n",
    "suffix": "Sample suffix: {psg_ids}",
}

# Sample invalid templates for testing validation
INVALID_SINGLETURN_TEMPLATES = [
    {"method": "pairwise", "body": "{rank} {candidate}"},  # Wrong method type
    {
        "method": "singleturn_listwise",
        "body": "Missing rank placeholder {rank}",
    },  # Missing required placeholder: {candidate}
    {
        "method": "singleturn_listwise",
        "body": "{rank} {candidate}",
        "unknown_key": "value",
    },  # Unknown key
    {
        "method": "singleturn_listwise",
        "prefix": "{num}",
        "body": "{rank} {candidate}",
        "suffix": "test",
    },  # Missing query placeholder in both prefix and suffix
]
INVALID_MULTITURN_TEMPLATES = [
    {
        "method": "singleturn_listwise",
        "body": "{rank} {candidate}",
        "body_assistant": "{rank}",
    },  # Wrong method type
    {
        "method": "multiturn_listwise",
        "body": "{rank} {candidate}",
        "body_assistant": "{rank}",
        "unknown_key": "value",
    },  # Unknown key
    {
        "method": "multiturn_listwise",
        "body": "{rank} {candidate}",
    },  # Missing assistant sections
    {
        "method": "multiturn_listwise",
        "prefix": "{num}",
        "body": "{rank} {candidate}",
        "body_assistant": "{rank}",
        "suffix": "test",
    },  # Missing prefix_assistant when body_assistant and prefix are both present
    {
        "method": "multiturn_listwise",
        "system_message": "You are a helpful assistant than ranks documents.",
        "prefix": "Sample prefix: Rank these {num} passages",
        "prefix_assistant": "Okay, please provide the passages.",
        "body": "[{rank}] {candidate}",
        "body_assistant": "Received passage [{rank}].",
        "suffix": "Sample suffix: Rank the provided {num} passages",
    },  # Missing query placeholder in both prefix and suffix
]


class TestListwiseInferenceHandler(unittest.TestCase):
    def test_listwise_valid_template_initialization(self):
        singleturn_listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )
        multiturn_listwise_inference_handler_1 = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        multiturn_listwise_inference_handler_2 = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_2
        )
        self.assertEqual(
            singleturn_listwise_inference_handler.template, VALID_SINGLETURN_TEMPLATE
        )
        self.assertEqual(
            multiturn_listwise_inference_handler_1.template, VALID_MULTITURN_TEMPLATE_1
        )
        self.assertEqual(
            multiturn_listwise_inference_handler_2.template, VALID_MULTITURN_TEMPLATE_2
        )

    def test_invalid_templates(self):
        for template in INVALID_SINGLETURN_TEMPLATES:
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    SingleTurnListwiseInferenceHandler(template)
        for template in INVALID_MULTITURN_TEMPLATES:
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    MultiTurnListwiseInferenceHandler(template)

    def test_prefix_generation(self):
        singleturn_listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )
        multiturn_listwise_inference_handler = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        (
            singleturn_prefix_text,
            _,
        ) = singleturn_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        (
            multiturn_prefix_text,
            _,
        ) = multiturn_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        expected_prefix_singleturn = (
            "Sample prefix: Rank these 1 passages for query: test query"
        )
        expected_prefix_multiturn = [
            {
                "role": "user",
                "content": "Sample prefix: Rank these 1 passages for query: test query",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

        self.assertEqual(singleturn_prefix_text, expected_prefix_singleturn)
        self.assertEqual(multiturn_prefix_text, expected_prefix_multiturn)

    def test_suffix_generation(self):
        singleturn_listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )
        lrl_inference_handler = SingleTurnListwiseInferenceHandler(VALID_LRL_TEMPLATE)
        multiturn_listwise_inference_handler = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        (
            _,
            singleturn_suffix_text,
        ) = singleturn_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        _, lrl_suffix_text = lrl_inference_handler._generate_prefix_suffix(
            1, "test query", rank_start=0, rank_end=2
        )
        (
            _,
            multiturn_suffix_text,
        ) = multiturn_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        expected_suffix = (
            "Sample suffix: Rank the provided 1 passages based on query: test query"
        )
        expected_lrl_suffix = "Sample suffix: [PASSAGE1, PASSAGE2]"

        self.assertEqual(singleturn_suffix_text, expected_suffix)
        self.assertEqual(lrl_suffix_text, expected_lrl_suffix)
        self.assertEqual(multiturn_suffix_text, expected_suffix)

    def test_body_generation_singleturn(self):
        listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )

        body_text_num = listwise_inference_handler._generate_body(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        expected_body_num = "[1] Title: Sample Title Content: Sample Text\n[2] Title: Sample Title Content: Sample Text\n"
        body_text_alpha = listwise_inference_handler._generate_body(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=True
        )
        expected_body_alpha = "[A] Title: Sample Title Content: Sample Text\n[B] Title: Sample Title Content: Sample Text\n"

        self.assertEqual(body_text_num, expected_body_num)
        self.assertEqual(body_text_alpha, expected_body_alpha)

    def test_body_generation_multiturn(self):
        listwise_inference_handler = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        body_text_singleturn = listwise_inference_handler._generate_body(
            r,
            rank_start=0,
            rank_end=2,
            max_length=6000,
            use_alpha=False,
            is_conversational=False,
        )
        body_text_multiturn = listwise_inference_handler._generate_body(
            r,
            rank_start=0,
            rank_end=2,
            max_length=6000,
            use_alpha=False,
            is_conversational=True,
        )
        expected_body_singleturn = "[1] Title: Sample Title Content: Sample Text[2] Title: Sample Title Content: Sample Text"
        expected_body_multiturn = [
            {"role": "user", "content": "[1] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [1]."},
            {"role": "user", "content": "[2] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [2]."},
        ]
        self.assertEqual(body_text_singleturn, expected_body_singleturn)
        self.assertEqual(body_text_multiturn, expected_body_multiturn)

    def test_generate_prompt_singleturn(self):
        listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )
        num_prompt = listwise_inference_handler.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        expected_prompt_num = [
            {"role": "system", "content": VALID_SINGLETURN_TEMPLATE["system_message"]},
            {
                "role": "user",
                "content": "Sample prefix: Rank these 2 passages for query: Sample Query[1] Title: Sample Title Content: Sample Text\n[2] Title: Sample Title Content: Sample Text\nSample suffix: Rank the provided 2 passages based on query: Sample Query",
            },
        ]
        alpha_prompt = listwise_inference_handler.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=True
        )
        expected_prompt_alpha = [
            {"role": "system", "content": VALID_SINGLETURN_TEMPLATE["system_message"]},
            {
                "role": "user",
                "content": "Sample prefix: Rank these 2 passages for query: Sample Query[A] Title: Sample Title Content: Sample Text\n[B] Title: Sample Title Content: Sample Text\nSample suffix: Rank the provided 2 passages based on query: Sample Query",
            },
        ]
        self.assertEqual(num_prompt, expected_prompt_num)
        self.assertEqual(alpha_prompt, expected_prompt_alpha)

    def test_generate_prompt_multiturn(self):
        listwise_inference_handler_1 = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        listwise_inference_handler_2 = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_2
        )
        prompt_1 = listwise_inference_handler_1.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        prompt_2 = listwise_inference_handler_2.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        expected_prompt_1 = [
            {"role": "system", "content": VALID_MULTITURN_TEMPLATE_1["system_message"]},
            {
                "role": "user",
                "content": "Sample prefix: Rank these 2 passages for query: Sample Query",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
            {"role": "user", "content": "[1] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [1]."},
            {"role": "user", "content": "[2] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [2]."},
            {
                "role": "user",
                "content": "Sample suffix: Rank the provided 2 passages based on query: Sample Query",
            },
        ]
        expected_prompt_2 = [
            {"role": "system", "content": VALID_MULTITURN_TEMPLATE_1["system_message"]},
            {
                "role": "user",
                "content": "Sample prefix: Rank these 2 passages for query: Sample Query",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
            {
                "role": "user",
                "content": "[1] Title: Sample Title Content: Sample Text\n[2] Title: Sample Title Content: Sample Text\nSample suffix: Rank the provided 2 passages based on query: Sample Query",
            },
        ]
        self.assertEqual(prompt_1, expected_prompt_1)
        self.assertEqual(prompt_2, expected_prompt_2)


if __name__ == "__main__":
    unittest.main()

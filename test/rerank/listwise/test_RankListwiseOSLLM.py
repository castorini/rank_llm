import unittest
from unittest.mock import MagicMock, patch

from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank import PromptMode
from rank_llm.rerank.listwise.listwise_conv_inference_handler import (
    ListwiseInferenceHandlerConv,
)
from rank_llm.rerank.listwise.listwise_norm_inference_handler import (
    ListwiseInferenceHandlerNorm,
)
from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM

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


VALID_NORM_TEMPLATE = {
    "method": "listwise_norm",
    "system_message": "You are a helpful assistant that ranks documents.",
    "prefix": "Sample prefix: Rank these {num} passages for query: {query}",
    "suffix": "Sample suffix: Rank the provided {num} passages based on query: {query}",
    "body": "[{rank}] {candidate}\n",
}
VALID_CONV_TEMPLATE = {
    "method": "listwise_conv",
    "system_message": "You are a helpful assistant than ranks documents.",
    "prefix": "Sample prefix: Rank these {num} passages for query: {query}",
    "prefix_assistant": "Okay, please provide the passages.",
    "body": "[{rank}] {candidate}",
    "body_assistant": "Received passage [{rank}].",
    "suffix": "Sample suffix: Rank the provided {num} passages based on query: {query}",
}

# Sample invalid templates for testing validation
INVALID_NORM_TEMPLATES = [
    {"method": "pairwise", "body": "{rank} {candidate}"},  # Wrong method type
    {
        "method": "listwise_norm",
        "body": "Missing rank placeholder {rank}",
    },  # Missing required placeholder: {candidate}
    {
        "method": "listwise_norm",
        "body": "{rank} {candidate}",
        "unknown_key": "value",
    },  # Unknown key
    {
        "method": "listwise_norm",
        "prefix": "{num}",
        "body": "{rank} {candidate}",
        "suffix": "test",
    },  # Missing query placeholder in both prefix and suffix
]
INVALID_CONV_TEMPLATES = [
    {
        "method": "listwise_norm",
        "body": "{rank} {candidate}",
        "body_assistant": "{rank}",
    },  # Wrong method type
    {
        "method": "listwise_conv",
        "body": "{rank} {candidate}",
        "body_assistant": "{rank}",
        "unknown_key": "value",
    },  # Unknown key
    {
        "method": "listwise_conv",
        "body": "{rank} {candidate}",
    },  # Missing assistant sections
    {
        "method": "listwise_conv",
        "prefix": "{num}",
        "body": "{rank} {candidate}",
        "body_assistant": "{rank}",
        "suffix": "test",
    },  # Missing prefix_assistant when body_assistant and prefix are both present
    {
        "method": "listwise_conv",
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
        norm_listwise_inference_handler = ListwiseInferenceHandlerNorm(
            VALID_NORM_TEMPLATE
        )
        conv_listwise_inference_handler = ListwiseInferenceHandlerConv(
            VALID_CONV_TEMPLATE
        )
        self.assertEqual(norm_listwise_inference_handler.template, VALID_NORM_TEMPLATE)
        self.assertEqual(conv_listwise_inference_handler.template, VALID_CONV_TEMPLATE)

    def test_invalid_templates(self):
        for template in INVALID_NORM_TEMPLATES:
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    ListwiseInferenceHandlerNorm(template)
        for template in INVALID_CONV_TEMPLATES:
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    ListwiseInferenceHandlerConv(template)

    def test_prefix_generation(self):
        norm_listwise_inference_handler = ListwiseInferenceHandlerNorm(
            VALID_NORM_TEMPLATE
        )
        conv_listwise_inference_handler = ListwiseInferenceHandlerConv(
            VALID_CONV_TEMPLATE
        )
        prefix_text_norm, _ = norm_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        prefix_text_conv, _ = conv_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        expected_prefix_norm = (
            "Sample prefix: Rank these 1 passages for query: test query"
        )
        expected_prefix_conv = [
            {
                "role": "user",
                "content": "Sample prefix: Rank these 1 passages for query: test query",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

        self.assertEqual(prefix_text_norm, expected_prefix_norm)
        self.assertEqual(prefix_text_conv, expected_prefix_conv)

    def test_suffix_generation(self):
        norm_listwise_inference_handler = ListwiseInferenceHandlerNorm(
            VALID_NORM_TEMPLATE
        )
        conv_listwise_inference_handler = ListwiseInferenceHandlerConv(
            VALID_CONV_TEMPLATE
        )
        _, norm_suffix_text = norm_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        _, conv_suffix_text = conv_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        expected_suffix = (
            "Sample suffix: Rank the provided 1 passages based on query: test query"
        )

        self.assertEqual(norm_suffix_text, expected_suffix)
        self.assertEqual(conv_suffix_text, expected_suffix)

    def test_body_generation_norm(self):
        listwise_inference_handler = ListwiseInferenceHandlerNorm(VALID_NORM_TEMPLATE)
        body_text = listwise_inference_handler._generate_body(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        expected_body = "[1] Title: Sample Title Content: Sample Text\n[2] Title: Sample Title Content: Sample Text\n"
        self.assertEqual(body_text, expected_body)

    def test_body_generation_conv(self):
        listwise_inference_handler = ListwiseInferenceHandlerConv(VALID_CONV_TEMPLATE)
        body_text_norm = listwise_inference_handler._generate_body(
            r,
            rank_start=0,
            rank_end=2,
            max_length=6000,
            use_alpha=False,
            is_conversational=False,
        )
        body_text_conv = listwise_inference_handler._generate_body(
            r,
            rank_start=0,
            rank_end=2,
            max_length=6000,
            use_alpha=False,
            is_conversational=True,
        )
        expected_body_norm = "[1] Title: Sample Title Content: Sample Text[2] Title: Sample Title Content: Sample Text"
        expected_body_conv = [
            {"role": "user", "content": "[1] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [1]."},
            {"role": "user", "content": "[2] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [2]."},
        ]
        self.assertEqual(body_text_norm, expected_body_norm)
        self.assertEqual(body_text_conv, expected_body_conv)

    def test_generate_prompt_norm(self):
        listwise_inference_handler = ListwiseInferenceHandlerNorm(VALID_NORM_TEMPLATE)
        prompt = listwise_inference_handler.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        expected_prompt = [
            {"role": "system", "content": VALID_NORM_TEMPLATE["system_message"]},
            {
                "role": "user",
                "content": "Sample prefix: Rank these 2 passages for query: Sample Query[1] Title: Sample Title Content: Sample Text\n[2] Title: Sample Title Content: Sample Text\nSample suffix: Rank the provided 2 passages based on query: Sample Query",
            },
        ]
        self.assertEqual(prompt, expected_prompt)

    def test_generate_prompt_conv(self):
        listwise_inference_handler = ListwiseInferenceHandlerConv(VALID_CONV_TEMPLATE)
        prompt = listwise_inference_handler.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        expected_prompt = [
            {"role": "system", "content": VALID_CONV_TEMPLATE["system_message"]},
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
        self.assertEqual(prompt, expected_prompt)


if __name__ == "__main__":
    unittest.main()

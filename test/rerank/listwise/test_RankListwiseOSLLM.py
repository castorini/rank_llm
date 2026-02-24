import unittest
from unittest.mock import MagicMock, patch

from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.rankllm import PromptMode

# model, context_size, prompt_template_path, num_few_shot_examples, variable_passages, window_size, system_message
valid_inputs = [
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
        0,
        True,
        10,
        "Default Message",
    ),
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
        0,
        False,
        10,
        "Default Message",
    ),
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
        0,
        True,
        10,
        "",
    ),
    (
        "castorini/rank_vicuna_7b_v1",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
        0,
        True,
        10,
        "",
    ),
    (
        "castorini/rank_vicuna_7b_v1_noda",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
        0,
        True,
        10,
        "",
    ),
    (
        "castorini/rank_vicuna_7b_v1_fp16",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
        0,
        True,
        10,
        "",
    ),
    (
        "castorini/rank_vicuna_7b_v1_noda_fp16",
        4096,
        "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
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

        # Mock Tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template.side_effect = (
            lambda messages, **kwargs: str(messages)
        )
        self.mock_tokenizer.encode.side_effect = lambda x, **kwargs: [0] * (
            len(x) // 4 + 1
        )

        # Patch the handlers used by RankListwiseOSLLM
        self.patcher_vllm_handler = patch(
            "rank_llm.rerank.listwise.rank_listwise_os_llm.VllmHandler"
        )
        self.patcher_openai_handler = patch(
            "rank_llm.rerank.listwise.rank_listwise_os_llm.VllmHandlerWithOpenAISDK"
        )

        self.mock_vllm_handler_class = self.patcher_vllm_handler.start()
        self.mock_openai_handler_class = self.patcher_openai_handler.start()

        self.mock_vllm_handler_instance = self.mock_vllm_handler_class.return_value
        self.mock_vllm_handler_instance.get_tokenizer.return_value = self.mock_tokenizer

        self.mock_openai_handler_instance = self.mock_openai_handler_class.return_value
        self.mock_openai_handler_instance.get_tokenizer.return_value = (
            self.mock_tokenizer
        )

    def tearDown(self):
        self.patcher_cuda.stop()
        self.patcher_vllm_handler.stop()
        self.patcher_openai_handler.stop()

    def test_init_with_vllm_generate(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            context_size=4096,
            window_size=10,
        )
        self.assertTrue(hasattr(model_coordinator, "_vllm_handler"))
        self.assertIsNone(model_coordinator._base_url)

    def test_init_with_openai_sdk(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            context_size=4096,
            window_size=10,
            base_url="http://localhost:8000/v1",
        )
        self.assertTrue(hasattr(model_coordinator, "_vllm_handler"))
        self.assertEqual(model_coordinator._base_url, "http://localhost:8000/v1")

    def test_valid_inputs(self):
        for (
            model,
            context_size,
            prompt_template_path,
            num_few_shot_examples,
            variable_passages,
            window_size,
            system_message,
        ) in valid_inputs:
            model_coordinator = RankListwiseOSLLM(
                model=model,
                context_size=context_size,
                prompt_template_path=prompt_template_path,
                num_few_shot_examples=num_few_shot_examples,
                variable_passages=variable_passages,
                window_size=window_size,
                system_message=system_message,
            )
            self.assertEqual(model_coordinator._model, model)
            self.assertEqual(model_coordinator._context_size, context_size)
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
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            name="rank_zephyr",
            context_size=4096,
            prompt_template_path="src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
            num_few_shot_examples=0,
            variable_passages=True,
            window_size=10,
            system_message="",
        )

        mock_num_output_tokens.return_value = 40
        output = model_coordinator.num_output_tokens()
        self.assertEqual(output, 40)

        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            name="rank_zephyr",
            context_size=4096,
            prompt_template_path="src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
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
            prompt_template_path="src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
            num_few_shot_examples=0,
            variable_passages=True,
            window_size=5,
            system_message="",
        )

        mock_run_llm_batched.return_value = [
            (
                "> [1] > [2] > [3] > [4] > [5",
                "one is more relevant than the others",
                {"prompt_tokens": 10, "completion_tokens": 19},
            )
        ]
        output, reasoning, usage = model_coordinator.run_llm_batched(
            [
                "How are you doing ? What is your name? What is your age? What is your favorite color?"
            ]
        )[0]
        expected_output = "> [1] > [2] > [3] > [4] > [5"
        self.assertEqual(output, expected_output)
        self.assertEqual(reasoning, "one is more relevant than the others")
        self.assertEqual(usage["completion_tokens"], 19)

    def test_create_prompt(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            name="rank_zephyr",
            context_size=4096,
            prompt_template_path="src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
            num_few_shot_examples=0,
            variable_passages=True,
            window_size=5,
            system_message="",
            device="cpu",
        )

        import re

        def get_first_int(s):
            if isinstance(s, list):
                s = next((m["content"] for m in s if m["role"] == "user"), "")
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
            prompt_template_path="src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
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


if __name__ == "__main__":
    unittest.main()

import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM

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
        # Patch device detection and torch so tests run without a GPU
        self.patcher_device = patch(
            "rank_llm.utils.default_device", return_value="cuda"
        )
        self.mock_device = self.patcher_device.start()
        self.patcher_torch = patch(
            "rank_llm.rerank.listwise.rank_listwise_os_llm.torch", new=MagicMock()
        )
        self.mock_torch = self.patcher_torch.start()
        self.mock_torch.cuda.is_available.return_value = True
        self.mock_torch.cuda.device_count.return_value = 1
        # Ensure the vllm guard does not fire
        self.patcher_vllm_mod = patch(
            "rank_llm.rerank.listwise.rank_listwise_os_llm.vllm", new=MagicMock()
        )
        self.patcher_vllm_mod.start()

        # Mock Tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template.side_effect = (
            lambda messages, **kwargs: str(messages)
        )
        self.mock_tokenizer.encode.side_effect = lambda x, **kwargs: (
            [0] * (len(x) // 4 + 1)
        )

        self.patcher_auto_tokenizer = patch(
            "rank_llm.rerank.listwise.rank_listwise_os_llm.AutoTokenizer"
        )
        self.mock_auto_tokenizer = self.patcher_auto_tokenizer.start()
        self.mock_auto_tokenizer.from_pretrained.return_value = self.mock_tokenizer

        # Patch the handlers used by RankListwiseOSLLM without importing optional deps.
        self.mock_vllm_handler_class = MagicMock()
        self.mock_openai_handler_class = MagicMock()
        self.patcher_handler_modules = patch.dict(
            "sys.modules",
            {
                "rank_llm.rerank.vllm_handler": SimpleNamespace(
                    VllmHandler=self.mock_vllm_handler_class
                ),
                "rank_llm.rerank.vllm_handler_with_openai_sdk": SimpleNamespace(
                    VllmHandlerWithOpenAISDK=self.mock_openai_handler_class
                ),
            },
        )
        self.patcher_handler_modules.start()

        self.mock_vllm_handler_instance = self.mock_vllm_handler_class.return_value
        self.mock_vllm_handler_instance.get_tokenizer.return_value = self.mock_tokenizer

        self.mock_openai_handler_instance = self.mock_openai_handler_class.return_value
        self.mock_openai_handler_instance.get_tokenizer.return_value = (
            self.mock_tokenizer
        )

    def tearDown(self):
        self.patcher_device.stop()
        self.patcher_torch.stop()
        self.patcher_vllm_mod.stop()
        self.patcher_auto_tokenizer.stop()
        self.patcher_handler_modules.stop()

    def test_init_with_vllm_generate(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            context_size=4096,
            window_size=10,
        )
        self.assertTrue(hasattr(model_coordinator, "_vllm_handler"))
        self.assertIsNone(model_coordinator._base_url)

    def test_close_delegates_to_vllm_handler(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            context_size=4096,
            window_size=10,
        )

        model_coordinator.close()

        self.mock_vllm_handler_instance.close.assert_called_once_with()

    def test_vllm_max_model_len_covers_prompt_and_output_budgets(self):
        cases = (
            ("castorini/rank_zephyr_7b_v1_full", False, 10000, 4096),
            ("castorini/rank_vicuna_7b_v1", False, 10000, 4096),
            ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", True, 30000, 34096),
        )
        for model, is_thinking, reasoning_token_budget, expected in cases:
            with self.subTest(model=model, is_thinking=is_thinking):
                self.mock_vllm_handler_class.reset_mock()
                RankListwiseOSLLM(
                    model=model,
                    context_size=4096,
                    window_size=20,
                    is_thinking=is_thinking,
                    reasoning_token_budget=reasoning_token_budget,
                )

                self.assertEqual(
                    self.mock_vllm_handler_class.call_args.kwargs["max_model_len"],
                    expected,
                )

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

    def test_use_logits_ranks_first_token_logprobs(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/first_mistral",
            window_size=20,
            use_logits=True,
            use_alpha=True,
            sampling_kwargs={"temperature": 0.7, "top_p": 0.9},
        )
        first_token_logprobs = {
            1: SimpleNamespace(decoded_token="A", logprob=-1.0),
            2: SimpleNamespace(decoded_token="B", logprob=-0.1),
            3: SimpleNamespace(decoded_token="A", logprob=-0.2),
            4: SimpleNamespace(decoded_token="C", logprob=0.0),
            5: SimpleNamespace(decoded_token="not-an-id", logprob=1.0),
        }
        self.mock_vllm_handler_instance.generate_logprobs_async = AsyncMock(
            return_value=([first_token_logprobs], 17, 1)
        )

        llm_output = asyncio.run(
            model_coordinator.run_llm_async("rendered prompt", current_window_size=2)
        )

        self.assertEqual(
            llm_output,
            (
                "[B] > [A]",
                "",
                {
                    "prompt_tokens": 17,
                    "completion_tokens": 1,
                    "total_tokens": 18,
                },
            ),
        )
        self.mock_vllm_handler_instance.generate_logprobs_async.assert_awaited_once_with(
            prompt="rendered prompt[",
            min_tokens=1,
            max_tokens=1,
            logprobs=30,
            sampling_extra={"temperature": 0.0, "top_p": 0.9},
        )

        rerank_input = copy.copy(r)
        rerank_input.candidates = list(r.candidates)
        rerank_input.invocations_history = []
        reranked = model_coordinator._apply_llm_output_to_result(
            rerank_input,
            llm_output,
            "rendered prompt",
            16,
            0,
            2,
            populate_invocations_history=True,
        )
        self.assertEqual(
            [candidate.docid for candidate in reranked.candidates[:2]], ["d2", "d1"]
        )
        self.assertEqual(len(reranked.invocations_history), 1)
        invocation = reranked.invocations_history[0]
        self.assertEqual(invocation.input_token_count, 17)
        self.assertEqual(invocation.output_token_count, 1)
        self.assertEqual(invocation.token_usage, llm_output[2])

    def test_run_llm_async_without_logits_uses_text_generation(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            window_size=20,
            sampling_kwargs={"top_p": 0.9},
        )
        self.mock_vllm_handler_instance.generate_output_async = AsyncMock(
            return_value=("[1] > [2]", 12, 5)
        )
        self.mock_vllm_handler_instance.generate_logprobs_async = AsyncMock()

        llm_output = asyncio.run(
            model_coordinator.run_llm_async("rendered prompt", current_window_size=2)
        )

        self.assertEqual(
            llm_output,
            (
                "[1] > [2]",
                "",
                {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                },
            ),
        )
        self.mock_vllm_handler_instance.generate_output_async.assert_awaited_once()
        self.mock_vllm_handler_instance.generate_logprobs_async.assert_not_awaited()

    def test_use_logits_numeric_ids_respect_current_window(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/first_mistral",
            window_size=20,
            use_logits=True,
            use_alpha=False,
        )
        permutation, evaluations = model_coordinator._evaluate_logits(
            {
                1: SimpleNamespace(decoded_token="1", logprob=-0.5),
                2: SimpleNamespace(decoded_token="2", logprob=-0.1),
                3: SimpleNamespace(decoded_token="3", logprob=0.0),
            },
            (1, 2),
        )

        self.assertEqual(permutation, "[2] > [1]")
        self.assertEqual(evaluations, {1: -0.5, 2: -0.1})

    def test_use_logits_requires_local_vllm(self):
        with self.assertRaisesRegex(
            ValueError, "only supported by the in-process vLLM backend"
        ):
            RankListwiseOSLLM(
                model="castorini/first_mistral",
                use_logits=True,
                base_url="http://localhost:8000/v1",
            )

    def test_use_logits_requires_first_token_logprobs(self):
        model_coordinator = RankListwiseOSLLM(
            model="castorini/first_mistral",
            use_logits=True,
            use_alpha=True,
        )
        self.mock_vllm_handler_instance.generate_logprobs_async = AsyncMock(
            return_value=(None, 10, 1)
        )

        with self.assertRaisesRegex(
            RuntimeError, "did not return first-token logprobs"
        ):
            asyncio.run(model_coordinator.run_llm_async("rendered prompt"))

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

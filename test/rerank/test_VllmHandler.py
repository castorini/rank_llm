import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestVllmHandler(unittest.TestCase):
    def setUp(self):
        self.patcher_async_engine_args = patch("vllm.AsyncEngineArgs")
        self.patcher_async_llm_engine = patch("vllm.AsyncLLMEngine")
        self.patcher_sampling = patch("vllm.SamplingParams")
        self.patcher_atexit = patch("atexit.register")

        self.mock_engine_args_class = self.patcher_async_engine_args.start()
        self.mock_engine_class = self.patcher_async_llm_engine.start()
        self.mock_sampling_params_class = self.patcher_sampling.start()
        self.patcher_atexit.start()

        self.mock_engine_instance = self.mock_engine_class.from_engine_args.return_value
        self.mock_tokenizer = MagicMock()
        # get_tokenizer() on AsyncLLMEngine is a coroutine
        self.mock_engine_instance.get_tokenizer = AsyncMock(
            return_value=self.mock_tokenizer
        )

        from rank_llm.rerank.vllm_handler import VllmHandler

        self.handler = VllmHandler(
            model="test-model",
            download_dir="/tmp",
            enforce_eager=True,
            max_logprobs=5,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
        )

    def tearDown(self):
        self.patcher_async_engine_args.stop()
        self.patcher_async_llm_engine.stop()
        self.patcher_sampling.stop()
        self.patcher_atexit.stop()

    def test_init(self):
        self.mock_engine_class.from_engine_args.assert_called_once_with(
            self.mock_engine_args_class.return_value
        )
        self.assertEqual(self.handler._engine, self.mock_engine_instance)
        self.assertEqual(self.handler._model, "test-model")
        self.assertIsNone(self.handler._tokenizer)

    def test_get_tokenizer(self):
        tokenizer = self.handler.get_tokenizer()
        self.assertEqual(tokenizer, self.mock_tokenizer)
        # Second call should return cached tokenizer without calling engine again
        self.handler.get_tokenizer()
        self.mock_engine_instance.get_tokenizer.assert_awaited_once()

    def test_generate_output_async(self):
        mock_request_output = MagicMock()
        mock_request_output.finished = True
        mock_request_output.outputs = [MagicMock()]
        mock_request_output.outputs[0].text = "generated text"
        mock_request_output.outputs[0].token_ids = [1, 2, 3]
        mock_request_output.prompt_token_ids = [4, 5, 6, 7]

        async def fake_generate(prompt, sampling_params, request_id):
            yield mock_request_output

        self.mock_engine_instance.generate = fake_generate

        text, prompt_tokens, completion_tokens = asyncio.run(
            self.handler.generate_output_async(
                prompt="test prompt",
                min_tokens=1,
                max_tokens=10,
                temperature=0.0,
            )
        )

        self.assertEqual(text, "generated text")
        self.assertEqual(prompt_tokens, 4)
        self.assertEqual(completion_tokens, 3)

    def test_generate_output_async_sampling_params(self):
        mock_request_output = MagicMock()
        mock_request_output.finished = True
        mock_request_output.outputs = [MagicMock()]
        mock_request_output.outputs[0].text = ""
        mock_request_output.outputs[0].token_ids = []
        mock_request_output.prompt_token_ids = []

        async def fake_generate(prompt, sampling_params, request_id):
            yield mock_request_output

        self.mock_engine_instance.generate = fake_generate

        asyncio.run(
            self.handler.generate_output_async(
                prompt="test",
                min_tokens=5,
                max_tokens=20,
                temperature=0.7,
                logprobs=3,
            )
        )

        self.mock_sampling_params_class.assert_called_once_with(
            min_tokens=5,
            max_tokens=20,
            temperature=0.7,
            logprobs=3,
        )


if __name__ == "__main__":
    unittest.main()

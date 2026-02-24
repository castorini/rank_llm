import unittest
from unittest.mock import MagicMock, patch

from rank_llm.rerank.vllm_handler_with_openai_sdk import VllmHandlerWithOpenAISDK


class TestVllmHandlerWithOpenAISDK(unittest.TestCase):
    @patch("rank_llm.rerank.vllm_handler_with_openai_sdk.OpenAI")
    @patch("rank_llm.rerank.vllm_handler_with_openai_sdk.AutoTokenizer")
    @patch("rank_llm.rerank.vllm_handler_with_openai_sdk.ThreadPoolExecutor")
    def setUp(self, mock_executor_class, mock_tokenizer_class, mock_openai_class):
        self.mock_client = mock_openai_class.return_value
        self.mock_tokenizer = mock_tokenizer_class.from_pretrained.return_value
        self.mock_executor = mock_executor_class.return_value
        self.mock_executor_class = mock_executor_class

        # Mock models.list() for init when model is None
        mock_model_data = MagicMock()
        mock_model_data.id = "default-model"
        self.mock_client.models.list.return_value.data = [mock_model_data]

        self.handler = VllmHandlerWithOpenAISDK(
            base_url="http://localhost:8000/v1", model="test-model", batch_size=16
        )

    def test_init(self):
        self.assertEqual(self.handler._model, "test-model")
        self.assertEqual(self.handler._tokenizer, self.mock_tokenizer)
        self.mock_executor_class.assert_called_once_with(max_workers=16)

    def test_get_tokenizer(self):
        self.assertEqual(self.handler.get_tokenizer(), self.mock_tokenizer)

    def test_one_inference_success(self):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "test content"
        mock_choice.message.reasoning = "test reasoning"
        mock_response.choices = [mock_choice]
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 10}

        self.mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "hello"}]
        text, reasoning, usage = self.handler._one_inference(messages, temperature=0)

        self.mock_client.chat.completions.create.assert_called_once_with(
            model="test-model", messages=messages, temperature=0
        )
        self.assertEqual(text, "test content")
        self.assertEqual(reasoning, "test reasoning")
        self.assertEqual(usage, {"prompt_tokens": 10})

    def test_one_inference_failure(self):
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")

        messages = [{"role": "user", "content": "hello"}]
        text, reasoning, usage = self.handler._one_inference(messages)

        self.assertEqual(text, "API Error")
        self.assertEqual(reasoning, "")
        self.assertEqual(usage, {})

    def test_chat_completions_uses_persistent_executor(self):
        self.mock_executor.map.return_value = iter([("t1", "r1", {}), ("t2", "r2", {})])

        prompts = [
            [{"role": "user", "content": "p1"}],
            [{"role": "user", "content": "p2"}],
        ]
        results = self.handler.chat_completions(prompts)

        self.mock_executor.map.assert_called_once()
        self.assertEqual(len(results), 2)

    def test_concurrency_execution_order(self):
        # Real ThreadPoolExecutor test to ensure results are returned in order
        import time
        from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor

        # Use a real executor for this specific test to verify ordering
        self.handler._executor = RealThreadPoolExecutor(max_workers=2)

        def mock_create(model, messages, **kwargs):
            # Simulate variable latency
            if "p1" in messages[0]["content"]:
                time.sleep(0.1)
                content = "r1"
            else:
                content = "r2"

            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = content
            mock_resp.choices[0].message.reasoning = ""
            mock_resp.usage.model_dump.return_value = {}
            return mock_resp

        self.mock_client.chat.completions.create.side_effect = mock_create

        prompts = [
            [{"role": "user", "content": "p1"}],
            [{"role": "user", "content": "p2"}],
        ]

        # Even though p1 takes longer, results should be in order [r1, r2]
        results = self.handler.chat_completions(prompts)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "r1")
        self.assertEqual(results[1][0], "r2")

    def test_partial_failures_in_batch(self):
        # Test that one failing request doesn't stop the whole batch
        from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor

        self.handler._executor = RealThreadPoolExecutor(max_workers=3)

        def mock_create(model, messages, **kwargs):
            if "fail" in messages[0]["content"]:
                raise Exception("Specific Error")

            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "success"
            mock_resp.choices[0].message.reasoning = ""
            mock_resp.usage.model_dump.return_value = {}
            return mock_resp

        self.mock_client.chat.completions.create.side_effect = mock_create

        prompts = [
            [{"role": "user", "content": "success"}],
            [{"role": "user", "content": "fail"}],
            [{"role": "user", "content": "success"}],
        ]

        results = self.handler.chat_completions(prompts)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "success")
        self.assertEqual(results[1][0], "Specific Error")
        self.assertEqual(results[2][0], "success")


if __name__ == "__main__":
    unittest.main()

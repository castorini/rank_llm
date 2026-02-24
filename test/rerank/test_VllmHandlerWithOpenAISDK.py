import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rank_llm.rerank.vllm_handler_with_openai_sdk import VllmHandlerWithOpenAISDK


class TestVllmHandlerWithOpenAISDK(unittest.TestCase):
    @patch("rank_llm.rerank.vllm_handler_with_openai_sdk.AsyncOpenAI")
    @patch("rank_llm.rerank.vllm_handler_with_openai_sdk.OpenAI")
    @patch("rank_llm.rerank.vllm_handler_with_openai_sdk.AutoTokenizer")
    def setUp(self, mock_tokenizer_class, mock_openai_class, mock_async_openai_class):
        self.mock_sync_client = mock_openai_class.return_value
        self.mock_async_client = mock_async_openai_class.return_value
        self.mock_tokenizer = mock_tokenizer_class.from_pretrained.return_value

        mock_model_data = MagicMock()
        mock_model_data.id = "default-model"
        self.mock_sync_client.models.list.return_value.data = [mock_model_data]

        self.handler = VllmHandlerWithOpenAISDK(
            base_url="http://localhost:8000/v1", model="test-model"
        )

    def test_init(self):
        self.assertEqual(self.handler._model, "test-model")
        self.assertEqual(self.handler._tokenizer, self.mock_tokenizer)

    def test_get_tokenizer(self):
        self.assertEqual(self.handler.get_tokenizer(), self.mock_tokenizer)

    def test_chat_completion_async_success(self):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "test content"
        mock_choice.message.reasoning = "test reasoning"
        mock_response.choices = [mock_choice]
        mock_response.usage.model_dump.return_value = {"prompt_tokens": 10}

        self.mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        messages = [{"role": "user", "content": "hello"}]
        text, reasoning, usage = asyncio.run(
            self.handler.chat_completion_async(messages, temperature=0)
        )

        self.mock_async_client.chat.completions.create.assert_called_once_with(
            model="test-model", messages=messages, temperature=0
        )
        self.assertEqual(text, "test content")
        self.assertEqual(reasoning, "test reasoning")
        self.assertEqual(usage, {"prompt_tokens": 10})

    def test_chat_completion_async_failure(self):
        self.mock_async_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        messages = [{"role": "user", "content": "hello"}]
        text, reasoning, usage = asyncio.run(
            self.handler.chat_completion_async(messages)
        )

        self.assertEqual(text, "API Error")
        self.assertEqual(reasoning, "")
        self.assertEqual(usage, {})

    def test_concurrent_requests_all_succeed(self):
        """All concurrent coroutines complete independently."""
        call_count = 0

        async def mock_create(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = messages[0]["content"]
            mock_resp.choices[0].message.reasoning = ""
            mock_resp.usage.model_dump.return_value = {}
            return mock_resp

        self.mock_async_client.chat.completions.create = mock_create

        prompts = [
            [{"role": "user", "content": "p1"}],
            [{"role": "user", "content": "p2"}],
            [{"role": "user", "content": "p3"}],
        ]

        async def run():
            return await asyncio.gather(
                *[self.handler.chat_completion_async(p) for p in prompts]
            )

        results = asyncio.run(run())

        self.assertEqual(call_count, 3)
        self.assertEqual(results[0][0], "p1")
        self.assertEqual(results[1][0], "p2")
        self.assertEqual(results[2][0], "p3")

    def test_partial_failures_in_batch(self):
        """A failing request returns the error string; others still succeed."""

        async def mock_create(model, messages, **kwargs):
            if "fail" in messages[0]["content"]:
                raise Exception("Specific Error")
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "success"
            mock_resp.choices[0].message.reasoning = ""
            mock_resp.usage.model_dump.return_value = {}
            return mock_resp

        self.mock_async_client.chat.completions.create = mock_create

        prompts = [
            [{"role": "user", "content": "success"}],
            [{"role": "user", "content": "fail"}],
            [{"role": "user", "content": "success"}],
        ]

        async def run():
            return await asyncio.gather(
                *[self.handler.chat_completion_async(p) for p in prompts]
            )

        results = asyncio.run(run())

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "success")
        self.assertEqual(results[1][0], "Specific Error")
        self.assertEqual(results[2][0], "success")


if __name__ == "__main__":
    unittest.main()

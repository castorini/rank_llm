import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from dacite import from_dict

from rank_llm.data import Request
from rank_llm.rerank.pointwise.pointwise_vllm import PointwiseVLLM


class _MockTokenizer:
    """Minimal tokenizer: word-split documents; yes/no token ids by string form."""

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        t = text.strip()
        if t in ("yes", "Yes", "YES"):
            return [101]
        if t in ("no", "No", "NO"):
            return [102]
        max_length = kwargs.get("max_length")
        toks = text.split()
        if max_length is not None:
            toks = toks[: int(max_length)]
        return toks

    def decode(self, tokens, skip_special_tokens: bool = True, **kwargs):
        return " ".join(str(t) for t in tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ):
        n = sum(len(m.get("content", "")) for m in messages)
        return [0] * max(8, min(n // 2 + 4, 500))


def _sample_request() -> Request:
    return from_dict(
        data_class=Request,
        data={
            "query": {"text": "weather today", "qid": "q1"},
            "candidates": [
                {
                    "doc": {"contents": "low relevance"},
                    "docid": "d1",
                    "score": 0.9,
                },
                {
                    "doc": {"contents": "high relevance answer"},
                    "docid": "d2",
                    "score": 0.8,
                },
            ],
        },
    )


class TestPointwiseVLLM(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_vllm = MagicMock()
        self.mock_vllm.get_tokenizer.return_value = _MockTokenizer()
        self.mock_vllm.chat_completion_score_async = AsyncMock(
            side_effect=lambda msgs, **kw: (
                "yes",
                1,
                0.9 if "high" in msgs[-1]["content"] else 0.1,
                {"prompt_tokens": 10, "completion_tokens": 1},
            )
        )

    def _make_reranker(self) -> PointwiseVLLM:
        with patch(
            "rank_llm.rerank.pointwise.pointwise_vllm.VllmHandlerWithOpenAISDK",
            return_value=self.mock_vllm,
        ):
            return PointwiseVLLM(
                model="dummy-model",
                base_url="http://127.0.0.1:8000/v1",
                batch_size=8,
                max_concurrent_llm_calls=4,
            )

    def test_rerank_batch_orders_by_score(self):
        r = self._make_reranker()
        out = r.rerank_batch([_sample_request()], rank_end=10)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].candidates[0].docid, "d2")
        self.assertEqual(out[0].candidates[1].docid, "d1")

    async def test_rerank_batch_async_concurrency_cap(self):
        """Overlapping async work respects semaphore."""
        in_flight = 0
        max_in_flight = 0

        async def slow_score(msgs, **kw):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.03)
            in_flight -= 1
            rel = 0.7 if "high" in msgs[-1]["content"] else 0.2
            return "yes", 1, rel, {"prompt_tokens": 5, "completion_tokens": 1}

        self.mock_vllm.chat_completion_score_async = AsyncMock(side_effect=slow_score)

        r = self._make_reranker()
        r._max_concurrent = 2
        r._llm_concurrency_sem = None

        req = _sample_request()

        async def one_batch():
            return await r.rerank_batch_async([req], rank_end=10)

        await asyncio.gather(one_batch(), one_batch())
        self.assertLessEqual(max_in_flight, 2)


if __name__ == "__main__":
    unittest.main()

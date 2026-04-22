import asyncio
import unittest
from unittest.mock import MagicMock, patch

from dacite import from_dict

from rank_llm.data import Request, Result
from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM


def _minimal_request() -> Request:
    return from_dict(
        data_class=Request,
        data={
            "query": {"text": "q", "qid": "q1"},
            "candidates": [
                {
                    "doc": {"contents": "a"},
                    "docid": "d1",
                    "score": 0.5,
                },
                {
                    "doc": {"contents": "b"},
                    "docid": "d2",
                    "score": 0.4,
                },
            ],
        },
    )


class TestRerankAsyncSharedConcurrency(unittest.IsolatedAsyncioTestCase):
    """Concurrent rerank_async calls share one LLM slot cap (listwise + vLLM path)."""

    async def test_concurrent_rerank_async_shares_semaphore(self):
        with (
            patch("rank_llm.utils.default_device", return_value="cuda"),
            patch("rank_llm.rerank.listwise.rank_listwise_os_llm.torch", new=MagicMock())
            as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            with patch(
                "rank_llm.rerank.listwise.rank_listwise_os_llm.vllm", new=MagicMock()
            ):
                with patch("rank_llm.rerank.vllm_handler.VllmHandler") as vh:
                    with patch(
                        "rank_llm.rerank.vllm_handler_with_openai_sdk.VllmHandlerWithOpenAISDK"
                    ) as oa:
                        mock_tok = MagicMock()
                        mock_tok.apply_chat_template.side_effect = (
                            lambda m, **k: str(m)
                        )
                        mock_tok.encode.side_effect = lambda x, **k: [0] * 4
                        vh.return_value.get_tokenizer.return_value = mock_tok
                        oa.return_value.get_tokenizer.return_value = mock_tok
                        m = RankListwiseOSLLM(
                            model="m",
                            context_size=256,
                            window_size=2,
                            stride=1,
                            batch_size=1,
                        )
        m.create_prompt = MagicMock(  # type: ignore[method-assign]
            return_value=("[p]", 1)
        )
        m.receive_permutation = MagicMock(side_effect=lambda res, *_a, **_k: res)  # type: ignore[method-assign]
        in_flight = 0
        max_in_flight = 0
        leave_order: list[str] = []

        async def slow_llm(*_a, **_k):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return "1 2", "", {"prompt_tokens": 1, "completion_tokens": 1}

        m.run_llm_async = slow_llm  # type: ignore[method-assign]

        async def trace(tag: str):
            _ = await m.rerank_async(_minimal_request(), rank_end=2)
            leave_order.append(tag)

        await asyncio.gather(trace("a"), trace("b"))

        self.assertEqual(max_in_flight, 1)
        self.assertEqual(leave_order, ["a", "b"])


if __name__ == "__main__":
    unittest.main()

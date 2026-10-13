import asyncio
import inspect
import threading
import unittest
from enum import Enum
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

from rank_llm.data import Query, Result
from test.test_cli_mcp import FakeMCP

FASTMCP_AVAILABLE = find_spec("fastmcp") is not None
if FASTMCP_AVAILABLE:
    from fastmcp import Client, FastMCP

    from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM
    from rank_llm.server.mcp.tools import register_rankllm_tools


@unittest.skipUnless(FASTMCP_AVAILABLE, "fastmcp is required")
class TestMCPRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_real_dispatch_runs_batched_bridge_for_both_tools(self):
        async def inference(prompt, window):
            return prompt, "", {"total_tokens": 1}

        def pipeline(**kwargs):
            # Exercise the real asyncio.run bridge without loading a GPU model.
            responses = RankListwiseOSLLM.run_llm_batched(
                SimpleNamespace(run_llm_async=inference), ["cats"], 1
            )
            self.assertEqual(responses[0][0], "cats")
            return [Result(query=Query(text="cats", qid="1"), candidates=[])]

        mcp = FastMCP("runtime-regression")
        register_rankllm_tools(mcp)
        with (
            patch("rank_llm.server.mcp.tools.run_mcp_rerank", side_effect=pipeline),
            patch(
                "rank_llm.server.mcp.tools.run_mcp_retrieve_and_rerank",
                side_effect=pipeline,
            ),
        ):
            async with Client(mcp) as client:
                for name, args in (
                    (
                        "rerank",
                        {"model_path": "model", "query_text": "cats", "candidates": []},
                    ),
                    ("retrieve_and_rerank", {"model_path": "model"}),
                ):
                    result = await client.call_tool(name, args)
                    self.assertFalse(result.is_error)
                    self.assertEqual(result.data[0].query.text, "cats")

    async def test_registered_schema_matches_wrapper_signature(self):
        fake = FakeMCP()
        register_rankllm_tools(fake)
        mcp = FastMCP("schema")
        register_rankllm_tools(mcp)
        async with Client(mcp) as client:
            for tool in await client.list_tools():
                signature = inspect.signature(fake.tools[tool.name])
                schema = tool.inputSchema
                self.assertEqual(set(schema["properties"]), set(signature.parameters))
                self.assertEqual(
                    set(schema.get("required", [])),
                    {
                        name
                        for name, p in signature.parameters.items()
                        if p.default is inspect.Parameter.empty
                    },
                )
                for name, p in signature.parameters.items():
                    if p.default is not inspect.Parameter.empty:
                        self.assertEqual(
                            schema["properties"][name]["default"],
                            p.default.value
                            if isinstance(p.default, Enum)
                            else p.default,
                        )

    async def test_worker_exception_is_preserved(self):
        mcp = FakeMCP()
        register_rankllm_tools(mcp)
        error = ValueError("invalid candidates")
        with patch("rank_llm.server.mcp.tools.run_mcp_rerank", side_effect=error):
            with self.assertRaises(ValueError) as raised:
                await mcp.tools["rerank"]("model", "cats", [])
        self.assertIs(raised.exception, error)

    async def test_cancellation_keeps_worker_serialized_and_loop_responsive(self):
        mcp = FakeMCP()
        register_rankllm_tools(mcp)
        started = threading.Event()
        release = threading.Event()
        second_started = threading.Event()

        def first(**kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release worker")
            return []

        def second(**kwargs):
            second_started.set()
            return []

        with (
            patch("rank_llm.server.mcp.tools.run_mcp_rerank", side_effect=first),
            patch(
                "rank_llm.server.mcp.tools.run_mcp_retrieve_and_rerank",
                side_effect=second,
            ),
        ):
            task = asyncio.create_task(mcp.tools["rerank"]("model", "cats", []))
            next_task = None
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                next_task = asyncio.create_task(
                    mcp.tools["retrieve_and_rerank"]("model")
                )
                # This timer can fire only if inference leaves the event loop responsive.
                await asyncio.sleep(0.05)
                self.assertFalse(second_started.is_set())
            finally:
                release.set()
                if next_task is not None:
                    await asyncio.wait_for(next_task, 5)
            self.assertTrue(second_started.is_set())

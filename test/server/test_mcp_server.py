import asyncio
import os
import sys
import unittest

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.tests import run_server_async

from rank_llm.retrieve import RetrievalMethod


def _make_mcp_server():
    """Build a MCP server with RankLLM tools."""
    from rank_llm.server.mcp.tools import register_rankllm_tools

    mcp = FastMCP("rankllm")
    register_rankllm_tools(mcp)
    return mcp


class TestMCPServer(unittest.TestCase):
    """Test MCP server via HTTP transport using FastMCP Client."""

    # Suppress server output during tests, comment out for debugging
    def setUp(self):
        self._devnull = open(os.devnull, "w")
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr
        sys.stdout = self._devnull
        sys.stderr = self._devnull

    def tearDown(self):
        sys.stdout = self._saved_stdout
        sys.stderr = self._saved_stderr
        self._devnull.close()

    def _run_async(self, coro):
        return asyncio.run(coro)

    async def _call_tool(self, name, arguments):
        # Create server inside this event loop so server._started is bound to it
        mcp = _make_mcp_server()
        async with run_server_async(
            mcp,
            transport="streamable-http",
        ) as url:
            async with Client(StreamableHttpTransport(url)) as client:
                return await client.call_tool(name, arguments)

    def test_retrieve_and_rerank_tool(self):
        # Cap Java heap so Pyserini/Anserini do not use all system RAM (GPU is fine).
        os.environ.setdefault("JAVA_OPTS", "-Xmx2g")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        response = self._run_async(
            self._call_tool(
                "retrieve_and_rerank",
                {
                    "model_path": "Qwen/Qwen3-0.6B",
                    "query": "cats",
                    "dataset": "nfc",
                    "output_jsonl_file": "temp/ranked_results.jsonl",
                    "output_trec_file": "temp/ranked_results.trec",
                    "retrieval_method": RetrievalMethod.BM25,
                    "top_k_candidates": 10,
                    "max_queries": 1,
                },
            )
        )
        self.assertFalse(response.is_error, msg=getattr(response, "content", response))
        self.assertIsNotNone(response.content)

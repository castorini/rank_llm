"""The MCP rerank tool accepts extra_fields and forwards them."""

import asyncio
import unittest
from unittest.mock import patch

try:
    from fastmcp import Client, FastMCP
except ImportError:  # optional dependency
    Client = FastMCP = None


@unittest.skipIf(FastMCP is None, "fastmcp is not installed")
class TestMcpRerankExtraFields(unittest.TestCase):
    def test_rerank_tool_declares_and_forwards_extra_fields(self):
        from rank_llm.server.mcp import tools

        mcp = FastMCP("test")
        tools.register_rankllm_tools(mcp)

        async def run():
            async with Client(mcp) as client:
                listed = {tool.name: tool for tool in await client.list_tools()}
                self.assertIn(
                    "extra_fields", listed["rerank"].inputSchema["properties"]
                )
                await client.call_tool(
                    "rerank",
                    {
                        "model_path": "m",
                        "query_text": "q",
                        "candidates": [],
                        "extra_fields": {"reasoning": "a trace"},
                    },
                )

        with patch.object(tools, "run_mcp_rerank", return_value=[]) as run_rerank:
            asyncio.run(run())
        self.assertEqual(
            run_rerank.call_args.kwargs["extra_fields"], {"reasoning": "a trace"}
        )


if __name__ == "__main__":
    unittest.main()

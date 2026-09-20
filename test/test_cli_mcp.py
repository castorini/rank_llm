import unittest
from importlib.util import find_spec
from unittest.mock import patch

FASTMCP_AVAILABLE = find_spec("fastmcp") is not None

if FASTMCP_AVAILABLE:
    from rank_llm.api.mcp.mcp_rankllm import run_mcp_server


@unittest.skipUnless(FASTMCP_AVAILABLE, "fastmcp is required for MCP tests")
class TestCLIMCP(unittest.TestCase):
    def test_run_mcp_server_dispatches_valid_transport_arguments(self):
        from fastmcp import FastMCP

        for transport in ("stdio", "http"):
            server = FastMCP("test")
            with (
                self.subTest(transport=transport),
                patch(
                    "rank_llm.api.mcp.mcp_rankllm.build_mcp_server", return_value=server
                ),
                patch.object(server, f"run_{transport}_async", autospec=True) as run,
            ):
                run_mcp_server(transport=transport, port=9000)
                expected = {"show_banner": True}
                if transport == "http":
                    expected.update(transport="http", port=9000)
                run.assert_awaited_once_with(**expected)


if __name__ == "__main__":
    unittest.main()

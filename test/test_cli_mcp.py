import unittest
from importlib.util import find_spec
from unittest.mock import patch

FASTMCP_AVAILABLE = find_spec("fastmcp") is not None

if FASTMCP_AVAILABLE:
    from rank_llm.api.cli.main import main
    from rank_llm.api.mcp.mcp_rankllm import run_mcp_server


@unittest.skipUnless(FASTMCP_AVAILABLE, "fastmcp is required for MCP tests")
class TestCLIMCP(unittest.TestCase):
    def test_run_mcp_server_dispatches_valid_transport_arguments(self):
        from fastmcp import FastMCP

        for argv, transport, port in (
            (None, "http", 8000),
            ([], "http", 8000),
            (["--transport", "stdio", "--port", "9000"], "stdio", 9000),
            (["--transport", "http", "--port", "9000"], "http", 9000),
        ):
            server = FastMCP("test")
            with (
                self.subTest(argv=argv),
                patch("rank_llm.api.cli.main.load_config", return_value=({}, None)),
                patch(
                    "rank_llm.api.mcp.mcp_rankllm.build_mcp_server", return_value=server
                ),
                patch.object(server, f"run_{transport}_async", autospec=True) as run,
            ):
                if argv is None:
                    run_mcp_server()
                else:
                    self.assertEqual(main(["serve", "mcp", *argv]), 0)
                expected = {"show_banner": True}
                if transport == "http":
                    expected.update(transport="http", port=port)
                run.assert_awaited_once_with(**expected)


if __name__ == "__main__":
    unittest.main()

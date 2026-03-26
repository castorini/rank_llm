import subprocess
import unittest
from importlib.util import find_spec
from pathlib import Path
from shutil import which
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
FASTMCP_AVAILABLE = find_spec("fastmcp") is not None

if FASTMCP_AVAILABLE:
    from rank_llm.server.mcp.mcp_rankllm import run_mcp_server
    from rank_llm.server.mcp.tools import register_rankllm_tools


class FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, description: str):
        del description

        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


@unittest.skipUnless(FASTMCP_AVAILABLE, "fastmcp is required for MCP tests")
class TestCLIMCP(unittest.TestCase):
    def test_register_rankllm_tools_uses_shared_rerank_handler(self):
        mcp = FakeMCP()
        with patch(
            "rank_llm.server.mcp.tools.run_mcp_rerank",
            return_value=[{"ok": True}],
        ) as mocked:
            register_rankllm_tools(mcp)
            result = mcp.tools["rerank"](
                model_path="model",
                query_text="cats",
                candidates=[{"docid": "1", "score": 1.0, "doc": "doc"}],
            )

        self.assertEqual(result, [{"ok": True}])
        self.assertEqual(mocked.call_args.kwargs["model_path"], "model")
        self.assertEqual(mocked.call_args.kwargs["query_text"], "cats")

    def test_register_rankllm_tools_uses_shared_retrieve_handler(self):
        mcp = FakeMCP()
        with patch(
            "rank_llm.server.mcp.tools.run_mcp_retrieve_and_rerank",
            return_value=[{"ok": True}],
        ) as mocked:
            register_rankllm_tools(mcp)
            result = mcp.tools["retrieve_and_rerank"](
                model_path="model",
                dataset="dl19",
            )

        self.assertEqual(result, [{"ok": True}])
        self.assertEqual(mocked.call_args.kwargs["model_path"], "model")
        self.assertEqual(mocked.call_args.kwargs["dataset"], "dl19")

    def test_run_mcp_server_uses_built_server(self):
        server = Mock()
        with patch(
            "rank_llm.server.mcp.mcp_rankllm.build_mcp_server",
            return_value=server,
        ) as mocked:
            run_mcp_server(transport="http", port=9000)

        mocked.assert_called_once_with()
        server.run.assert_called_once_with(transport="http", port=9000)

    def test_console_entrypoint_serve_mcp_help_resolves(self):
        cli = which("rank-llm")
        self.assertIsNotNone(cli, msg="rank-llm is not installed in PATH")

        help_result = subprocess.run(
            [cli, "serve", "mcp", "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(help_result.returncode, 0, msg=help_result.stderr)
        self.assertIn("Start the RankLLM MCP server", help_result.stdout)


if __name__ == "__main__":
    unittest.main()

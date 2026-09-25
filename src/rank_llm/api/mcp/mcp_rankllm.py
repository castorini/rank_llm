import sys

from rank_llm._optional import missing_extra_error
from rank_llm.api.cli.main import main as cli_main


def build_mcp_server():
    try:
        from fastmcp import FastMCP
        from pyserini.server.backend import get_backend
        from pyserini.server.mcp.tools import register_tools

        from rank_llm.api.mcp.tools import register_rankllm_tools
    except ImportError as exc:
        raise missing_extra_error(
            "mcp",
            "The MCP server requires FastMCP and Pyserini.",
        ) from exc

    mcp = FastMCP("rankllm")
    register_tools(mcp, get_backend())
    register_rankllm_tools(mcp)
    return mcp


def run_mcp_server(*, transport: str = "http", port: int = 8000):
    mcp = build_mcp_server()
    kwargs = {"port": port} if transport == "http" else {}
    mcp.run(transport=transport, **kwargs)


def main(argv=None):
    """Delegate legacy MCP arguments to the canonical CLI parser."""
    return cli_main(["serve", "mcp", *(sys.argv[1:] if argv is None else argv)])

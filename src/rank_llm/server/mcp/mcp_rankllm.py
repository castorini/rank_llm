import argparse

from fastmcp import FastMCP
from pyserini.server.mcp.tools import register_tools
from pyserini.server.search_controller import get_controller

from rank_llm.server.mcp.tools import register_rankllm_tools


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="MCPyserini Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode for the MCP server (default: stdio)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for HTTP transport (default: 8000)",
    )

    args = parser.parse_args()

    try:
        mcp = FastMCP("rankllm")

        # Pyserini tools
        register_tools(mcp, get_controller())
        # RankLLM tools
        register_rankllm_tools(mcp)

        mcp.run(transport=args.transport, port=args.port)

    except Exception as e:
        print("Error", e)
        raise

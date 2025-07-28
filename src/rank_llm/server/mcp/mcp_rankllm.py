import argparse

from fastmcp import FastMCP

from rank_llm.server.mcp.tools import register_rankllm_tools


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="MCPyserini Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport mode for the MCP server (default: stdio)",
    )

    args = parser.parse_args()

    try:
        mcp = FastMCP("rankllm")

        register_rankllm_tools(mcp)

        mcp.run(transport=args.transport)

    except Exception as e:
        print("Error", e)
        raise

import argparse

from rank_llm._optional import missing_extra_error
from rank_llm.cli.main import main as cli_main


def build_mcp_server():
    try:
        from fastmcp import FastMCP
        from pyserini.server.mcp.tools import register_tools
        from pyserini.server.search_controller import get_controller

        from rank_llm.server.mcp.tools import register_rankllm_tools
    except ImportError as exc:
        raise missing_extra_error(
            "server",
            "The MCP server requires FastMCP and Pyserini.",
        ) from exc

    mcp = FastMCP("rankllm")
    register_tools(mcp, get_controller())
    register_rankllm_tools(mcp)
    return mcp


def run_mcp_server(*, transport: str = "stdio", port: int = 8000):
    mcp = build_mcp_server()
    mcp.run(transport=transport, port=port)


def main(argv=None):
    if argv is not None:
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
        args = parser.parse_args(argv)
        return cli_main(
            [
                "serve",
                "mcp",
                "--transport",
                args.transport,
                "--port",
                str(args.port),
            ]
        )

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

    return cli_main(
        [
            "serve",
            "mcp",
            "--transport",
            args.transport,
            "--port",
            str(args.port),
        ]
    )

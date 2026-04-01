from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Literal

from rank_llm._optional import missing_extra_error
from rank_llm.cli.main import main as cli_main

if TYPE_CHECKING:
    from fastmcp import FastMCP

MCPTransport = Literal["stdio", "http", "sse", "streamable-http"]


def build_mcp_server() -> FastMCP[Any]:
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


def run_mcp_server(*, transport: MCPTransport = "stdio", port: int = 8000) -> None:
    mcp = build_mcp_server()
    mcp.run(transport=transport, port=port)


def main(argv: list[str] | None = None) -> int:
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

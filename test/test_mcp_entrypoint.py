import unittest
from unittest.mock import patch

from rank_llm.api.mcp import mcp_rankllm


class TestMCPEntrypoint(unittest.TestCase):
    def test_explicit_and_process_arguments_delegate_identically(self):
        for explicit in (True, False):
            with (
                self.subTest(explicit=explicit),
                patch(
                    "sys.argv", ["rankllm-mcp", "--transport", "http", "--port", "9000"]
                ),
                patch.object(mcp_rankllm, "cli_main", return_value=0) as cli,
            ):
                result = mcp_rankllm.main(
                    ["--transport", "http", "--port", "9000"] if explicit else None
                )
                self.assertEqual(result, 0)
                cli.assert_called_once_with(
                    ["serve", "mcp", "--transport", "http", "--port", "9000"]
                )

    def test_empty_explicit_arguments_use_defaults(self):
        with (
            patch("sys.argv", ["rankllm-mcp", "--port", "9000"]),
            patch.object(mcp_rankllm, "cli_main", return_value=0) as cli,
        ):
            self.assertEqual(mcp_rankllm.main([]), 0)
        cli.assert_called_once_with(
            ["serve", "mcp", "--transport", "stdio", "--port", "8000"]
        )

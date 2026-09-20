import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from shutil import which

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCLIPackaging(unittest.TestCase):
    def test_cli_module_imports(self):
        module = importlib.import_module("rank_llm.api.cli.main")
        self.assertTrue(callable(module.main))

    def test_server_mcp_module_imports(self):
        module = importlib.import_module("rank_llm.api.mcp")
        self.assertIsNotNone(module)

    def test_shared_api_and_cli_do_not_import_server_dependencies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import builtins
original_import = builtins.__import__
def without_servers(name, *args, **kwargs):
    if name.split('.')[0] in {'fastapi', 'fastmcp', 'pyserini'}:
        raise AssertionError(f'Unexpected server import: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_servers
import rank_llm.api.cli.main
import rank_llm.api.operations
import rank_llm.api.options
import rank_llm.api.capabilities
""",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_console_entrypoint_help_resolves(self):
        cli = which("rank-llm")
        self.assertIsNotNone(cli, msg="rank-llm is not installed in PATH")

        help_result = subprocess.run(
            [cli, "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, msg=help_result.stderr)
        self.assertIn("Packaged CLI entrypoint for RankLLM", help_result.stdout)


if __name__ == "__main__":
    unittest.main()

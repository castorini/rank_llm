import importlib
import subprocess
import unittest
from pathlib import Path
from shutil import which

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCLIPackaging(unittest.TestCase):
    def test_cli_module_imports(self):
        module = importlib.import_module("rank_llm.cli.main")
        self.assertTrue(callable(module.main))

    def test_server_mcp_module_imports(self):
        module = importlib.import_module("rank_llm.server.mcp")
        self.assertIsNotNone(module)

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

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rank_llm.cli.main import main


class TestCLIIntrospection(unittest.TestCase):
    def test_describe_returns_command_metadata(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--output", "json", "describe", "rerank"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        description = payload["artifacts"][0]["value"]
        self.assertEqual(description["name"], "rerank")
        self.assertIn("input_modes", description)

    def test_schema_returns_named_schema(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--output", "json", "schema", "cli-envelope"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        schema = payload["artifacts"][0]["value"]
        self.assertEqual(schema["name"], "cli-envelope")
        self.assertIn("required", schema["schema"])

    def test_rerank_direct_input_schema_includes_overrides(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--output", "json", "schema", "rerank-direct-input"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        schema = payload["artifacts"][0]["value"]["schema"]
        self.assertIn("overrides", schema["properties"])
        self.assertEqual(
            schema["properties"]["overrides"]["properties"]["reasoning_effort"]["enum"],
            ["low", "medium", "high"],
        )

    def test_doctor_returns_readiness(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--output", "json", "doctor"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        doctor = payload["artifacts"][0]["value"]
        self.assertIn("python_version", doctor)
        self.assertIn("overall_status", doctor)
        self.assertIn("config_file", doctor)

    def test_doctor_can_be_mocked_for_dependency_states(self):
        stdout = io.StringIO()
        with patch(
            "rank_llm.cli.main.doctor_report",
            return_value={
                "python_version": "3.11.0",
                "python_ok": True,
                "optional_dependencies": {},
                "command_readiness": {},
                "overall_status": "ready",
            },
        ):
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--output", "json", "doctor"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["artifacts"][0]["value"]["overall_status"], "ready")

    def test_doctor_reports_loaded_config_file(self):
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / ".rank-llm.toml"
            config_path.write_text(
                'base_url = "http://localhost:9000"\n', encoding="utf-8"
            )
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["--output", "json", "doctor"])
            finally:
                os.chdir(old_cwd)
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            payload["artifacts"][0]["value"]["config_file"],
            ".rank-llm.toml",
        )
        self.assertEqual(
            payload["resolved"]["config"]["base_url"], "http://localhost:9000"
        )


if __name__ == "__main__":
    unittest.main()

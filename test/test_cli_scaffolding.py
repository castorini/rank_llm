import contextlib
import io
import json
import unittest
from unittest.mock import patch

from rank_llm.cli.main import main
from rank_llm.cli.responses import CommandResponse


class ProviderError(Exception):
    pass


ProviderError.__module__ = "openai"


class TestCLIResponses(unittest.TestCase):
    def test_command_response_envelope(self):
        response = CommandResponse(command="doctor", warnings=["stub"])
        envelope = response.to_envelope()
        self.assertEqual(envelope["schema_version"], "castorini.cli.v1")
        self.assertEqual(envelope["repo"], "rank_llm")
        self.assertEqual(envelope["command"], "doctor")
        self.assertEqual(envelope["warnings"], ["stub"])


class TestCLIParserAndOutput(unittest.TestCase):
    def test_top_level_help_does_not_expose_suppress_sentinels(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual("", stderr.getvalue())
        self.assertNotIn("==SUPPRESS==", stdout.getvalue())

    def test_missing_command_text_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main([])
        self.assertEqual(exit_code, 2)
        self.assertIn("No command provided. Choose one of:", stderr.getvalue())
        self.assertEqual("", stdout.getvalue())

    def test_invalid_argument_json_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(["--output", "json", "bogus"])
        self.assertEqual(exit_code, 2)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")
        self.assertEqual(payload["errors"][0]["code"], "invalid_arguments")

    def test_invalid_argument_json_error_with_equals_form(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(["--output=json", "bogus"])
        self.assertEqual(exit_code, 2)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")
        self.assertEqual(payload["errors"][0]["code"], "invalid_arguments")

    def test_invalid_argument_uses_passed_argv_for_command_detection(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(["--output", "json", "doctor", "--bad"])
        self.assertEqual(exit_code, 2)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "doctor")
        self.assertEqual(payload["errors"][0]["code"], "invalid_arguments")

    def test_doctor_text_output(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(["doctor"])
        self.assertEqual(exit_code, 0)
        self.assertEqual("", stderr.getvalue())
        self.assertIn('"python_version"', stdout.getvalue())

    def test_unexpected_runtime_error_json_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("rank_llm.cli.main._run_command", side_effect=RuntimeError("boom")),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = main(["--output", "json", "doctor"])
        self.assertEqual(exit_code, 6)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "runtime_error")
        self.assertEqual(payload["errors"][0]["code"], "runtime_error")

    def test_unexpected_provider_error_json_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch(
                "rank_llm.cli.main._run_command",
                side_effect=ProviderError("Rate limit exceeded"),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = main(["--output", "json", "doctor"])
        self.assertEqual(exit_code, 6)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "provider_error")
        self.assertEqual(payload["errors"][0]["code"], "provider_error")
        self.assertTrue(payload["errors"][0]["retryable"])

    def test_unexpected_missing_prerequisite_json_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("rank_llm.cli.main._run_command", side_effect=ImportError("fastapi")),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = main(["--output", "json", "doctor"])
        self.assertEqual(exit_code, 3)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")
        self.assertEqual(payload["errors"][0]["code"], "missing_prerequisite")

    def test_assertion_error_without_prerequisite_markers_stays_runtime_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch(
                "rank_llm.cli.main._run_command",
                side_effect=AssertionError("tuple shape mismatch"),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = main(["--output", "json", "doctor"])
        self.assertEqual(exit_code, 6)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "runtime_error")
        self.assertEqual(payload["errors"][0]["code"], "runtime_error")

    def test_prerequisite_assertion_maps_to_missing_prerequisite(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch(
                "rank_llm.cli.main._run_command",
                side_effect=AssertionError(
                    "Ensure that `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION` are set"
                ),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = main(["--output", "json", "doctor"])
        self.assertEqual(exit_code, 3)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")
        self.assertEqual(payload["errors"][0]["code"], "missing_prerequisite")

    def test_analyze_can_report_partial_success(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        summary = {
            "files": ["responses.jsonl"],
            "verbose": False,
            "metrics": {"ok": 2, "wrong_format": 1, "repetition": 0},
        }
        with (
            patch(
                "rank_llm.cli.main.run_response_analysis_files", return_value=summary
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = main(
                ["--output", "json", "analyze", "--files", "responses.jsonl"]
            )
        self.assertEqual(exit_code, 7)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "partial_success")
        self.assertIn("mix of valid outputs", payload["warnings"][0])


if __name__ == "__main__":
    unittest.main()

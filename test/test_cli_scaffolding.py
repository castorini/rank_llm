import contextlib
import io
import json
import unittest

from rank_llm.cli.main import main
from rank_llm.cli.responses import CommandResponse


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


if __name__ == "__main__":
    unittest.main()

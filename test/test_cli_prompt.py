import contextlib
import io
import json
import unittest

from rank_llm.cli.main import main


class TestCLIPrompt(unittest.TestCase):
    def test_prompt_list_returns_catalog(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--output", "json", "prompt", "list"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        catalog = payload["artifacts"][0]["value"]
        self.assertTrue(
            any(entry["name"] == "rank_zephyr_template" for entry in catalog)
        )

    def test_prompt_show_returns_template(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                ["--output", "json", "prompt", "show", "rank_zephyr_template"]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        template = payload["artifacts"][0]["value"]["template"]
        self.assertEqual(template["method"], "singleturn_listwise")

    def test_prompt_render_returns_messages(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "prompt",
                    "render",
                    "rank_zephyr_template",
                    "--input-json",
                    '{"query":"cats","candidates":["doc one","doc two"]}',
                ]
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        messages = payload["artifacts"][0]["value"]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("cats", messages[-1]["content"])
        self.assertIn("[1] doc one", messages[-1]["content"])

    def test_prompt_show_text_output(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["prompt", "show", "rank_gpt_template"])
        self.assertEqual(exit_code, 0)
        text = stdout.getvalue()
        self.assertIn("method: multiturn_listwise", text)
        self.assertIn("[prefix_user]", text)

    def test_prompt_render_invalid_payload_fails(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "prompt",
                    "render",
                    "rank_zephyr_template",
                    "--input-json",
                    '{"query":"cats"}',
                ]
            )
        self.assertEqual(exit_code, 5)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "validation_error")


if __name__ == "__main__":
    unittest.main()

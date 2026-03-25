import contextlib
import io
import json
import unittest
from unittest.mock import patch

from rank_llm.cli.main import main


class TestCLIRerankCommand(unittest.TestCase):
    def test_rerank_dataset_mode_uses_retrieve_handler(self):
        stdout = io.StringIO()
        with patch(
            "rank_llm.cli.main.run_mcp_retrieve_and_rerank",
            return_value=[{"dataset": True}],
        ) as mocked, contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "rerank",
                    "--model-path",
                    "model",
                    "--dataset",
                    "dl19",
                    "--retrieval-method",
                    "bm25",
                    "--query",
                    "cats",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked.call_args.kwargs["query"], "cats")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "rerank")
        self.assertEqual(payload["inputs"]["mode"], "dataset")

    def test_rerank_requests_file_mode_uses_retrieve_handler(self):
        with patch(
            "rank_llm.cli.main.run_mcp_retrieve_and_rerank",
            return_value=[{"requests_file": True}],
        ) as mocked:
            exit_code = main(
                [
                    "rerank",
                    "--model-path",
                    "model",
                    "--requests-file",
                    "requests.jsonl",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked.call_args.kwargs["requests_file"], "requests.jsonl")

    def test_rerank_direct_json_mode_uses_inline_handler(self):
        payload = '{"query":"cats","candidates":["doc one"]}'
        stdout = io.StringIO()
        with patch(
            "rank_llm.cli.main.run_mcp_rerank",
            return_value=[{"direct": True}],
        ) as mocked, contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--output",
                    "json",
                    "rerank",
                    "--model-path",
                    "model",
                    "--input-json",
                    payload,
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked.call_args.kwargs["query_text"], "cats")
        self.assertEqual(mocked.call_args.kwargs["candidates"][0]["doc"], "doc one")
        envelope = json.loads(stdout.getvalue())
        self.assertEqual(envelope["artifacts"][0]["value"], [{"direct": True}])

    def test_rerank_stdin_mode_uses_inline_handler(self):
        stdout = io.StringIO()
        stdin = io.StringIO('{"query":{"text":"cats","qid":"q1"},"candidates":[{"text":"doc"}]}')
        with patch(
            "rank_llm.cli.main.run_mcp_rerank",
            return_value=[{"stdin": True}],
        ) as mocked, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()), patch(
            "sys.stdin",
            stdin,
        ):
            exit_code = main(["--output", "json", "rerank", "--model-path", "model", "--stdin"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked.call_args.kwargs["query_id"], "q1")
        self.assertEqual(mocked.call_args.kwargs["candidates"][0]["doc"], "doc")


if __name__ == "__main__":
    unittest.main()

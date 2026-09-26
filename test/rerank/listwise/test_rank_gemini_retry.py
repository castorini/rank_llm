import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rank_llm.rerank.listwise import rank_gemini


class FakeAPIError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(str(code))


class TestGeminiInferenceRetry(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.responses = []

        def make_client(api_key):
            def generate_content(**kwargs):
                self.calls.append((api_key, kwargs))
                response = self.responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return SimpleNamespace(text=response)

            return SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content)
            )

        self.patches = (
            patch.object(rank_gemini, "genai", SimpleNamespace(Client=make_client)),
            patch.object(
                rank_gemini,
                "errors",
                SimpleNamespace(APIError=FakeAPIError),
            ),
            patch.object(rank_gemini.time, "sleep"),
        )
        for active_patch in self.patches:
            started = active_patch.start()
            self.addCleanup(active_patch.stop)
            if active_patch is self.patches[-1]:
                self.sleep = started

        self.reranker = rank_gemini.SafeGenai.__new__(rank_gemini.SafeGenai)
        self.reranker._model = "gemini-test"
        self.reranker._request_config = {}
        self.reranker._keys = ["first", "second"]
        self.reranker._cur_key_id = 0
        self.reranker._set_client()

    def test_retries_transient_error_then_succeeds(self):
        self.responses = [FakeAPIError(503), "ranking"]

        self.assertEqual(
            self.reranker._call_inference("prompt", return_text=True), "ranking"
        )
        self.assertEqual([key for key, _ in self.calls], ["first", "second"])
        self.sleep.assert_called_once_with(1.0)

    def test_exhausts_retryable_error_after_three_attempts(self):
        self.responses = [FakeAPIError(429) for _ in range(3)]

        with self.assertRaises(FakeAPIError) as exc:
            self.reranker._call_inference("prompt")

        self.assertEqual(exc.exception.code, 429)
        self.assertEqual([key for key, _ in self.calls], ["first", "second", "first"])
        self.assertEqual(self.sleep.call_count, 2)

    def test_permanent_api_error_surfaces_without_retry(self):
        for status in (400, 401, 404):
            with self.subTest(status=status):
                self.calls.clear()
                self.responses = [FakeAPIError(status)]
                with self.assertRaises(FakeAPIError):
                    self.reranker._call_inference("prompt")
                self.assertEqual(len(self.calls), 1)
        self.sleep.assert_not_called()

    def test_non_api_error_surfaces_without_retry(self):
        self.responses = [ValueError("bad request")]

        with self.assertRaises(ValueError):
            self.reranker._call_inference("prompt")

        self.assertEqual(len(self.calls), 1)
        self.sleep.assert_not_called()

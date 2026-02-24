import unittest
from unittest.mock import MagicMock, patch


# We patch vllm.LLM and vllm.SamplingParams to avoid actual model loading
class TestVllmHandler(unittest.TestCase):
    def setUp(self):
        # Patch vllm.LLM and vllm.SamplingParams
        self.patcher_llm = patch("vllm.LLM")
        self.patcher_sampling = patch("vllm.SamplingParams")

        self.mock_llm_class = self.patcher_llm.start()
        self.mock_sampling_params_class = self.patcher_sampling.start()

        self.mock_vllm_instance = self.mock_llm_class.return_value
        self.mock_tokenizer = MagicMock()
        self.mock_vllm_instance.get_tokenizer.return_value = self.mock_tokenizer

        # Now import VllmHandler after patching
        from rank_llm.rerank.vllm_handler import VllmHandler

        self.handler = VllmHandler(
            model="test-model",
            download_dir="/tmp",
            enforce_eager=True,
            max_logprobs=5,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
        )

    def tearDown(self):
        self.patcher_llm.stop()
        self.patcher_sampling.stop()

    def test_init(self):
        self.assertEqual(self.handler._vllm, self.mock_vllm_instance)
        self.assertEqual(self.handler._tokenizer, self.mock_tokenizer)

    def test_get_tokenizer(self):
        self.assertEqual(self.handler.get_tokenizer(), self.mock_tokenizer)

    def test_generate_output(self):
        prompts = ["prompt1", "prompt2"]
        mock_output = [MagicMock()]
        self.mock_vllm_instance.generate.return_value = mock_output

        result = self.handler.generate_output(
            prompts=prompts, min_tokens=1, max_tokens=10, temperature=0.7, logprobs=5
        )

        self.mock_sampling_params_class.assert_called_once_with(
            min_tokens=1, max_tokens=10, temperature=0.7, logprobs=5
        )
        self.mock_vllm_instance.generate.assert_called_once_with(
            prompts, self.mock_sampling_params_class.return_value
        )
        self.assertEqual(result, mock_output)


if __name__ == "__main__":
    unittest.main()

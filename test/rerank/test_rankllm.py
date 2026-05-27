import unittest
import warnings

from rank_llm.rerank.rankllm import PromptMode, RankLLM

# A real template so RankLLM.__init__ can build an inference handler.
_TEMPLATE = "src/rank_llm/rerank/prompt_templates/rank_gpt_template.yaml"


class _MinimalRankLLM(RankLLM):
    """Smallest concrete RankLLM subclass so the abstract base can be
    instantiated in tests. The abstract methods are never exercised here;
    only __init__ behavior (the prompt_mode deprecation) is under test.
    """

    def run_llm_batched(self, prompts, **kwargs):
        return []

    def run_llm(self, prompt, **kwargs):
        return "", 0

    def create_prompt_batched(self, results, rank_start, rank_end):
        return []

    def create_prompt(self, result, rank_start, rank_end):
        return "", 0

    def get_num_tokens(self, prompt):
        return 0

    def cost_per_1k_token(self, input_token):
        return 0.0

    def num_output_tokens(self):
        return 0

    def rerank_batch(
        self,
        requests,
        rank_start=0,
        rank_end=100,
        shuffle_candidates=False,
        logging=False,
        **kwargs,
    ):
        return []

    def get_output_filename(
        self, top_k_candidates, dataset_name, shuffle_candidates, **kwargs
    ):
        return ""


class TestPromptModeDeprecation(unittest.TestCase):
    def _make(self, **kwargs):
        return _MinimalRankLLM(
            model="dummy",
            context_size=4096,
            prompt_template_path=_TEMPLATE,
            **kwargs,
        )

    def test_prompt_mode_emits_deprecation_warning(self):
        with self.assertWarns(DeprecationWarning) as ctx:
            self._make(prompt_mode=PromptMode.RANK_GPT)
        self.assertIn("PromptMode is deprecated", str(ctx.warning))

    def test_no_prompt_mode_emits_no_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._make()  # prompt_mode defaults to None
        prompt_mode_warnings = [
            w for w in caught if "PromptMode is deprecated" in str(w.message)
        ]
        self.assertEqual(prompt_mode_warnings, [])


if __name__ == "__main__":
    unittest.main()

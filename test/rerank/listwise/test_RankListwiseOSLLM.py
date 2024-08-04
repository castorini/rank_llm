import unittest
from unittest.mock import MagicMock, patch

from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank import PromptMode
from rank_llm.rerank.listwise import RankListwiseOSLLM

# model, context_size, prompt_mode, num_few_shot_examples, variable_passages, window_size, system_message
valid_inputs = [
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.RANK_GPT,
        0,
        True,
        10,
        "Default Message",
    ),
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.RANK_GPT,
        0,
        False,
        10,
        "Default Message",
    ),
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.RANK_GPT,
        0,
        True,
        30,
        "Default Message",
    ),
    ("castorini/rank_zephyr_7b_v1_full", 4096, PromptMode.RANK_GPT, 0, True, 10, ""),
    ("castorini/rank_vicuna_7b_v1", 4096, PromptMode.RANK_GPT, 0, True, 10, ""),
    ("castorini/rank_vicuna_7b_v1_noda", 4096, PromptMode.RANK_GPT, 0, True, 10, ""),
    ("castorini/rank_vicuna_7b_v1_fp16", 4096, PromptMode.RANK_GPT, 0, True, 10, ""),
    (
        "castorini/rank_vicuna_7b_v1_noda_fp16",
        4096,
        PromptMode.RANK_GPT,
        0,
        True,
        10,
        "",
    ),
]

failure_inputs = [
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_zephyr_7b_v1_full",
        4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1",
        4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_noda",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_noda",
        4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_fp16",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_fp16",
        4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_noda_fp16",
        4096,
        PromptMode.UNSPECIFIED,
        0,
        True,
        30,
        "Default Message",
    ),
    (
        "castorini/rank_vicuna_7b_v1_noda_fp16",
        4096,
        PromptMode.LRL,
        0,
        True,
        30,
        "Default Message",
    ),
]


r = from_dict(
    data_class=Result,
    data={
        "query": {"text": "Sample Query", "qid": "q1"},
        "candidates": [
            {
                "doc": {
                    "contents": "Title: Sample Title Content: Sample Text",
                },
                "docid": "d1",
                "score": 0.5,
            },
            {
                "doc": {
                    "contents": "Title: Sample Title Content: Sample Text",
                },
                "docid": "d2",
                "score": 0.4,
            },
            {
                "doc": {
                    "contents": "Title: Sample Title Content: Sample Text",
                },
                "docid": "d3",
                "score": 0.4,
            },
            {
                "doc": {
                    "contents": "Title: Sample Title Content: Sample Text",
                },
                "docid": "d4",
                "score": 0.3,
            },
        ],
    },
)


class TestRankListwiseOSLLM(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("rank_llm.rerank.rank_listwise_os_llm.load_model")
        self.mock_load_model = self.patcher.start()
        self.mock_llm = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_load_model.return_value = self.mock_llm, self.mock_tokenizer

        self.patcher_cuda = patch("torch.cuda.is_available")
        self.mock_cuda = self.patcher_cuda.start()
        self.mock_cuda.return_value = True

    def tearDown(self):
        self.patcher.stop()
        self.patcher_cuda.stop()

    def test_valid_inputs(self):
        for (
            model,
            context_size,
            prompt_mode,
            num_few_shot_examples,
            variable_passages,
            window_size,
            system_message,
        ) in valid_inputs:
            agent = RankListwiseOSLLM(
                model=model,
                context_size=context_size,
                prompt_mode=prompt_mode,
                num_few_shot_examples=num_few_shot_examples,
                variable_passages=variable_passages,
                window_size=window_size,
                system_message=system_message,
            )
            self.assertEqual(agent._model, model)
            self.assertEqual(agent._context_size, context_size)
            self.assertEqual(agent._prompt_mode, prompt_mode)
            self.assertEqual(agent._num_few_shot_examples, num_few_shot_examples)
            self.assertEqual(agent._variable_passages, variable_passages)
            self.assertEqual(agent._window_size, window_size)
            self.assertEqual(agent._system_message, system_message)

    def test_failure_inputs(self):
        for (
            model,
            context_size,
            prompt_mode,
            num_few_shot_examples,
            variable_passages,
            window_size,
            system_message,
        ) in failure_inputs:
            with self.assertRaises(ValueError):
                agent = RankListwiseOSLLM(
                    model=model,
                    context_size=context_size,
                    prompt_mode=prompt_mode,
                    num_few_shot_examples=num_few_shot_examples,
                    variable_passages=variable_passages,
                    window_size=window_size,
                    system_message=system_message,
                )

    @patch("rank_llm.rerank.rank_listwise_os_llm.RankListwiseOSLLM.num_output_tokens")
    def test_num_output_tokens(self, mock_num_output_tokens):
        # Creating PyseriniRetriever instance
        agent = RankListwiseOSLLM(
            "castorini/rank_zephyr_7b_v1_full",
            4096,
            PromptMode.RANK_GPT,
            0,
            variable_passages=True,
            window_size=10,
            system_message="",
        )

        mock_num_output_tokens.return_value = 40
        output = agent.num_output_tokens()
        self.assertEqual(output, 40)

        # print(output)
        agent = RankListwiseOSLLM(
            "castorini/rank_vicuna_7b_v1",
            4096,
            PromptMode.RANK_GPT,
            0,
            variable_passages=True,
            window_size=5,
            system_message="",
        )

        mock_num_output_tokens.return_value = 19
        output = agent.num_output_tokens()
        self.assertEqual(output, 19)

    @patch("rank_llm.rerank.rank_listwise_os_llm.RankListwiseOSLLM.run_llm")
    def test_run_llm(self, mock_run_llm):
        agent = RankListwiseOSLLM(
            "castorini/rank_zephyr_7b_v1_full",
            4096,
            PromptMode.RANK_GPT,
            0,
            variable_passages=True,
            window_size=5,
            system_message="",
        )

        mock_run_llm.return_value = ("> [1] > [2] > [3] > [4] > [5", 19)
        output, size = agent.run_llm(
            "How are you doing ? What is your name? What is your age? What is your favorite color?"
        )
        expected_output = "> [1] > [2] > [3] > [4] > [5"
        self.assertEqual(output, expected_output)
        self.assertEqual(size, len([char for char in output if char != " "]))

    def test_create_prompt(
        self,
    ):
        agent = RankListwiseOSLLM(
            "castorini/rank_zephyr_7b_v1_full",
            4096,
            PromptMode.RANK_GPT,
            0,
            variable_passages=True,
            window_size=5,
            system_message="",
            device="cpu",
        )

        import re

        def get_first_int(s):
            match = re.search(r"\d+", s)
            return int(match.group()) if match else None

        start_end_pairs = [(1, 3), (2, 4), (3, 5), (5, 6)]
        for start, end in start_end_pairs:
            prompt, length = agent.create_prompt(r, start, end)
            expected_output = min(end, len(r.candidates)) - max(0, start)
            self.assertEqual(get_first_int(prompt), max(expected_output, 0))

    @patch("rank_llm.rerank.rank_listwise_os_llm.RankListwiseOSLLM.get_num_tokens")
    def test_get_num_tokens(self, mock_get_num_tokens):
        agent = RankListwiseOSLLM(
            "castorini/rank_zephyr_7b_v1_full",
            4096,
            PromptMode.RANK_GPT,
            0,
            variable_passages=True,
            window_size=5,
            system_message="",
            device="cpu",
        )

        mock_get_num_tokens.return_value = 22
        output = agent.get_num_tokens(
            "How are you doing? What is your name? What is your age? What is your favorite color?"
        )
        self.assertEqual(output, 22)


if __name__ == "__main__":
    unittest.main()

import unittest

from rank_llm.rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.result import Result

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


r = Result(
    query="Sample Query",
    hits=[
        {
            "content": "Title: Sample Title Content: Sample Text",
            "qid": None,
            "docid": "d1",
            "rank": 1,
            "score": 0.5,
        },
        {
            "content": "Title: Sample Title Content: Sample Text",
            "qid": None,
            "docid": "d2",
            "rank": 2,
            "score": 0.4,
        },
        {
            "content": "Title: Sample Title Content: Sample Text",
            "qid": None,
            "docid": "d3",
            "rank": 3,
            "score": 0.4,
        },
        {
            "content": "Title: Sample Title Content: Sample Text",
            "qid": None,
            "docid": "d4",
            "rank": 4,
            "score": 0.3,
        },
    ],
)


class TestRankListwiseOSLLM(unittest.TestCase):
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

    def test_num_output_tokens(self):
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

        output = agent.num_output_tokens()
        self.assertEqual(output, 19)

    def test_run_llm(self):
        agent = RankListwiseOSLLM(
            "castorini/rank_zephyr_7b_v1_full",
            4096,
            PromptMode.RANK_GPT,
            0,
            variable_passages=True,
            window_size=5,
            system_message="",
        )
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
        )

        import re

        def get_first_int(s):
            match = re.search(r"\d+", s)
            return int(match.group()) if match else None

        start_end_pairs = [(1, 3), (2, 4), (3, 5), (5, 6)]
        for start, end in start_end_pairs:
            prompt, length = agent.create_prompt(r, start, end)
            expected_output = min(end, len(r.hits)) - max(0, start)
            self.assertEqual(get_first_int(prompt), max(expected_output, 0))

    def test_get_num_tokens(self):
        agent = RankListwiseOSLLM(
            "castorini/rank_zephyr_7b_v1_full",
            4096,
            PromptMode.RANK_GPT,
            0,
            variable_passages=True,
            window_size=5,
            system_message="",
        )

        output = agent.get_num_tokens(
            "How are you doing? What is your name? What is your age? What is your favorite color?"
        )
        self.assertEqual(output, 22)


class TestRankListwiseOSLLMBatching(unittest.TestCase):
    def test_sliding_windows_batched(self):
        # mock docs
        mock_results = [
            (
                "query1",
                Result(
                    hits=[
                        {"content": "doc1 query1", "score": 0.9},
                        {"content": "doc2 query1", "score": 0.8},
                    ]
                ),
            ),
            (
                "query2",
                Result(
                    hits=[
                        {"content": "doc3 query2", "score": 0.7},
                        {"content": "doc4 query2", "score": 0.6},
                        {"content": "doc5 query2", "score": 0.5},
                    ]
                ),
            ),
        ]

        agent = RankListwiseOSLLM(
            model="castorini/rank_zephyr_7b_v1_full",
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            variable_passages=True,
            window_size=2,
            system_message="",
        )

        batched_results = agent.sliding_windows_batched(
            queries_documents=mock_results,
            window_size=2,
            step=1,
            shuffle_candidates=False,
            logging=True,
        )

        self.assertEqual(len(batched_results), 2, "Should process two queries")

        for result in batched_results:
            self.assertTrue(
                isinstance(result, Result), "result must be a 'Result' object."
            )
            self.assertTrue(len(result.hits) > 0, "result.hits size must be > 0.")
            for hit in result.hits:
                self.assertIn("content", hit, "hit must contain 'content' key")
                self.assertIn("score", hit, "hit must contain 'score' key")


if __name__ == "__main__":
    unittest.main()

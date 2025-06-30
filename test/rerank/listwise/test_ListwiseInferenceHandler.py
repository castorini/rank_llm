import unittest

from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank.listwise.multiturn_listwise_inference_handler import (
    MultiTurnListwiseInferenceHandler,
)
from rank_llm.rerank.listwise.rankfid_inference_handler import RankFIDInferenceHandler
from rank_llm.rerank.listwise.singleturn_listwise_inference_handler import (
    SingleTurnListwiseInferenceHandler,
)

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

VALID_SINGLETURN_TEMPLATE = {
    "method": "singleturn_listwise",
    "system_message": "You are a helpful assistant that ranks documents.",
    "prefix": "Sample prefix: Rank these {num} passages for query: {query}",
    "suffix": "Sample suffix: Rank the provided {num} passages based on query: {query}",
    "body": "[{rank}] {candidate}\n",
    "outptut_patterns": ["test, test"],
}
VALID_RANKFID_TEMPLATE = {
    "method": "rankfid",
    "query": "question: {query}",
    "text": "question: {query} context: {passage} index: {index}",
    "outptut_patterns": ["test, test"],
}
VALID_LRL_TEMPLATE = {
    "method": "singleturn_listwise",
    "prefix": "Sample prefix: {query}",
    "body": "[{rank}] {candidate}\n",
    "suffix": "Sample suffix: {psg_ids}",
    "outptut_patterns": ["test, test"],
}
VALID_MULTITURN_TEMPLATE_1 = {
    "method": "multiturn_listwise",
    "system_message": "You are a helpful assistant than ranks documents.",
    "prefix_user": "Sample prefix: Rank these {num} passages for query: {query}",
    "prefix_assistant": "Okay, please provide the passages.",
    "body_user": "[{rank}] {candidate}",
    "body_assistant": "Received passage [{rank}].",
    "suffix_user": "Sample suffix: Rank the provided {num} passages based on query: {query}",
    "outptut_patterns": ["test, test"],
}
VALID_MULTITURN_TEMPLATE_2 = {
    "method": "multiturn_listwise",
    "system_message": "You are a helpful assistant than ranks documents.",
    "prefix_user": "Sample prefix: Rank these {num} passages for query: {query}",
    "prefix_assistant": "Okay, please provide the passages.",
    "body_user": "[{rank}] {candidate}\n",
    "suffix_user": "Sample suffix: Rank the provided {num} passages based on query: {query}",
    "outptut_patterns": ["test, test"],
}

# Sample invalid templates for testing validation
INVALID_SINGLETURN_TEMPLATES = [
    {
        "method": "pairwise",
        "body": "{rank} {candidate}",
        "outptut_patterns": ["test, test"],
    },  # Wrong method type
    {
        "method": "singleturn_listwise",
        "body": "Missing rank placeholder {rank}",
        "outptut_patterns": ["test, test"],
    },  # Missing required placeholder: {candidate}
    {
        "method": "singleturn_listwise",
        "body": "{rank} {candidate}",
        "outptut_patterns": ["test, test"],
        "unknown_key": "value",
    },  # Unknown key
    {
        "method": "singleturn_listwise",
        "prefix": "{num}",
        "body": "{rank} {candidate}",
        "suffix": "test",
        "outptut_patterns": ["test, test"],
    },  # Missing query placeholder in both prefix and suffix
]
INVALID_MULTITURN_TEMPLATES = [
    {
        "method": "singleturn_listwise",
        "body": "{rank} {candidate}",
        "body_assistant": "{rank}",
        "outptut_patterns": ["test, test"],
    },  # Wrong method type
    {
        "method": "multiturn_listwise",
        "body": "{rank} {candidate}",
        "body_assistant": "{rank}",
        "unknown_key": "value",
        "outptut_patterns": ["test, test"],
    },  # Unknown key
    {
        "method": "multiturn_listwise",
        "body": "{rank} {candidate}",
        "outptut_patterns": ["test, test"],
    },  # Missing assistant sections
    {
        "method": "multiturn_listwise",
        "prefix": "{num}",
        "body": "{rank} {candidate}",
        "body_assistant": "{rank}",
        "suffix": "test",
        "outptut_patterns": ["test, test"],
    },  # Missing prefix_assistant when body_assistant and prefix are both present
    {
        "method": "multiturn_listwise",
        "system_message": "You are a helpful assistant than ranks documents.",
        "prefix": "Sample prefix: Rank these {num} passages",
        "prefix_assistant": "Okay, please provide the passages.",
        "body": "[{rank}] {candidate}",
        "body_assistant": "Received passage [{rank}].",
        "suffix": "Sample suffix: Rank the provided {num} passages",
        "outptut_patterns": ["test, test"],
    },  # Missing query placeholder in both prefix and suffix
]
INVALID_RANKFID_TEMPLATES = [
    {
        "method": "singleturn_listwise",
        "text": "{query} {passage}",
        "outptut_patterns": ["test, test"],
    },  # Wrong method type
    {
        "method": "rankfid_listwise",
        "query": "question: {query}",
        "outptut_patterns": ["test, test"],
    },  # Missing text section
    {
        "method": "rankfid_listwise",
        "text": "question: {query} context: {passage}",
        "unknown_key": "value",
        "outptut_patterns": ["test, test"],
    },  # Unknown key
    {
        "method": "rankfid_listwise",
        "text": "context: {passage}",
        "outptut_patterns": ["test, test"],
    },  # Missing query placeholder
]


class TestListwiseInferenceHandler(unittest.TestCase):
    def test_listwise_valid_template_initialization(self):
        singleturn_listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )
        multiturn_listwise_inference_handler_1 = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        multiturn_listwise_inference_handler_2 = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_2
        )
        self.assertEqual(
            singleturn_listwise_inference_handler.template, VALID_SINGLETURN_TEMPLATE
        )
        self.assertEqual(
            multiturn_listwise_inference_handler_1.template, VALID_MULTITURN_TEMPLATE_1
        )
        self.assertEqual(
            multiturn_listwise_inference_handler_2.template, VALID_MULTITURN_TEMPLATE_2
        )

    def test_invalid_templates(self):
        for template in INVALID_SINGLETURN_TEMPLATES:
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    SingleTurnListwiseInferenceHandler(template)
        for template in INVALID_MULTITURN_TEMPLATES:
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    MultiTurnListwiseInferenceHandler(template)

    def test_prefix_generation(self):
        singleturn_listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )
        multiturn_listwise_inference_handler = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        (
            singleturn_prefix_text,
            _,
        ) = singleturn_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        (
            multiturn_prefix_text,
            _,
        ) = multiturn_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        expected_prefix_singleturn = (
            "Sample prefix: Rank these 1 passages for query: test query"
        )
        expected_prefix_multiturn = [
            {
                "role": "user",
                "content": "Sample prefix: Rank these 1 passages for query: test query",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

        self.assertEqual(singleturn_prefix_text, expected_prefix_singleturn)
        self.assertEqual(multiturn_prefix_text, expected_prefix_multiturn)

    def test_suffix_generation(self):
        singleturn_listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )
        lrl_inference_handler = SingleTurnListwiseInferenceHandler(VALID_LRL_TEMPLATE)
        multiturn_listwise_inference_handler = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        (
            _,
            singleturn_suffix_text,
        ) = singleturn_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        _, lrl_suffix_text = lrl_inference_handler._generate_prefix_suffix(
            1, "test query", rank_start=0, rank_end=2
        )
        (
            _,
            multiturn_suffix_text,
        ) = multiturn_listwise_inference_handler._generate_prefix_suffix(
            1, "test query"
        )
        expected_suffix = (
            "Sample suffix: Rank the provided 1 passages based on query: test query"
        )
        expected_lrl_suffix = "Sample suffix: [PASSAGE1, PASSAGE2]"

        self.assertEqual(singleturn_suffix_text, expected_suffix)
        self.assertEqual(lrl_suffix_text, expected_lrl_suffix)
        self.assertEqual(multiturn_suffix_text, expected_suffix)

    def test_body_generation_singleturn(self):
        listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )
        body_text_num = listwise_inference_handler._generate_body(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        expected_body_num = "[1] Title: Sample Title Content: Sample Text\n[2] Title: Sample Title Content: Sample Text\n"
        body_text_alpha = listwise_inference_handler._generate_body(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=True
        )
        expected_body_alpha = "[A] Title: Sample Title Content: Sample Text\n[B] Title: Sample Title Content: Sample Text\n"
        self.assertEqual(body_text_num, expected_body_num)
        self.assertEqual(body_text_alpha, expected_body_alpha)

    def test_body_generation_multiturn(self):
        listwise_inference_handler = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        body_text_singleturn = listwise_inference_handler._generate_body(
            r,
            rank_start=0,
            rank_end=2,
            max_length=6000,
            use_alpha=False,
            is_conversational=False,
        )
        body_text_multiturn = listwise_inference_handler._generate_body(
            r,
            rank_start=0,
            rank_end=2,
            max_length=6000,
            use_alpha=False,
            is_conversational=True,
        )
        expected_body_singleturn = "[1] Title: Sample Title Content: Sample Text[2] Title: Sample Title Content: Sample Text"
        expected_body_multiturn = [
            {"role": "user", "content": "[1] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [1]."},
            {"role": "user", "content": "[2] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [2]."},
        ]
        self.assertEqual(body_text_singleturn, expected_body_singleturn)
        self.assertEqual(body_text_multiturn, expected_body_multiturn)

    def test_generate_prompt_singleturn(self):
        listwise_inference_handler = SingleTurnListwiseInferenceHandler(
            VALID_SINGLETURN_TEMPLATE
        )
        num_prompt = listwise_inference_handler.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        expected_prompt_num = [
            {"role": "system", "content": VALID_SINGLETURN_TEMPLATE["system_message"]},
            {
                "role": "user",
                "content": "Sample prefix: Rank these 2 passages for query: Sample Query[1] Title: Sample Title Content: Sample Text\n[2] Title: Sample Title Content: Sample Text\nSample suffix: Rank the provided 2 passages based on query: Sample Query",
            },
        ]
        alpha_prompt = listwise_inference_handler.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=True
        )
        expected_prompt_alpha = [
            {"role": "system", "content": VALID_SINGLETURN_TEMPLATE["system_message"]},
            {
                "role": "user",
                "content": "Sample prefix: Rank these 2 passages for query: Sample Query[A] Title: Sample Title Content: Sample Text\n[B] Title: Sample Title Content: Sample Text\nSample suffix: Rank the provided 2 passages based on query: Sample Query",
            },
        ]
        self.assertEqual(num_prompt, expected_prompt_num)
        self.assertEqual(alpha_prompt, expected_prompt_alpha)

    def test_generate_prompt_multiturn(self):
        listwise_inference_handler_1 = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_1
        )
        listwise_inference_handler_2 = MultiTurnListwiseInferenceHandler(
            VALID_MULTITURN_TEMPLATE_2
        )
        prompt_1 = listwise_inference_handler_1.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        prompt_2 = listwise_inference_handler_2.generate_prompt(
            r, rank_start=0, rank_end=2, max_length=6000, use_alpha=False
        )
        expected_prompt_1 = [
            {"role": "system", "content": VALID_MULTITURN_TEMPLATE_1["system_message"]},
            {
                "role": "user",
                "content": "Sample prefix: Rank these 2 passages for query: Sample Query",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
            {"role": "user", "content": "[1] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [1]."},
            {"role": "user", "content": "[2] Title: Sample Title Content: Sample Text"},
            {"role": "assistant", "content": "Received passage [2]."},
            {
                "role": "user",
                "content": "Sample suffix: Rank the provided 2 passages based on query: Sample Query",
            },
        ]
        expected_prompt_2 = [
            {"role": "system", "content": VALID_MULTITURN_TEMPLATE_1["system_message"]},
            {
                "role": "user",
                "content": "Sample prefix: Rank these 2 passages for query: Sample Query",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
            {
                "role": "user",
                "content": "[1] Title: Sample Title Content: Sample Text\n[2] Title: Sample Title Content: Sample Text\nSample suffix: Rank the provided 2 passages based on query: Sample Query",
            },
        ]
        self.assertEqual(prompt_1, expected_prompt_1)
        self.assertEqual(prompt_2, expected_prompt_2)


class TestRankFIDInferenceHandler(unittest.TestCase):
    def test_rankfid_valid_template_initialization(self):
        rankfid_inference_handler = RankFIDInferenceHandler(VALID_RANKFID_TEMPLATE)
        self.assertEqual(rankfid_inference_handler.template, VALID_RANKFID_TEMPLATE)

    def test_rankfid_invalid_templates(self):
        for template in INVALID_RANKFID_TEMPLATES:
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    RankFIDInferenceHandler(template)

    def test_query_generation(self):
        rankfid_inference_handler = RankFIDInferenceHandler(VALID_RANKFID_TEMPLATE)
        query = rankfid_inference_handler._generate_query("test query")
        expected_query = "question: test query"
        self.assertEqual(query, expected_query)

    def test_text_generation(self):
        rankfid_listwise_inference_handler = RankFIDInferenceHandler(
            VALID_RANKFID_TEMPLATE
        )
        prompts = rankfid_listwise_inference_handler.generate_prompt(
            result=r, rank_start=0, rank_end=2, max_tokens=6000
        )
        expected_prompts = [
            {
                "query": "question: Sample Query",
                "text": "question: Sample Query context: Title: Sample Title Content: Sample Text index: 1",
            },
            {
                "query": "question: Sample Query",
                "text": "question: Sample Query context: Title: Sample Title Content: Sample Text index: 2",
            },
        ]
        self.assertEqual(prompts, expected_prompts)


if __name__ == "__main__":
    unittest.main()

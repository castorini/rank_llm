"""Where extra fields may be used, which names they may take, and how empty values render."""

import unittest

from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank.listwise.multiturn_listwise_inference_handler import (
    MultiTurnListwiseInferenceHandler,
)
from rank_llm.rerank.listwise.singleturn_listwise_inference_handler import (
    SingleTurnListwiseInferenceHandler,
)
from rank_llm.rerank.pairwise.pairwise_inference_handler import (
    PairwiseInferenceHandler,
)
from rank_llm.rerank.pointwise.pointwise_inference_handler import (
    PointwiseInferenceHandler,
)
from test.rerank.conftest import FakeTokenizer

LISTWISE = {
    "method": "singleturn_listwise",
    "prefix": "Query: {query}\n",
    "body": "[{rank}] {candidate}\n",
    "output_validation_regex": r"^\[\d+\]",
    "output_extraction_regex": r"\[(\d+)\]",
}


def result(extra_fields=None):
    return from_dict(
        data_class=Result,
        data={
            "query": {
                "text": "the query",
                "qid": "q1",
                "extra_fields": extra_fields or {},
            },
            "candidates": [
                {"doc": {"contents": "A passage."}, "docid": "d1", "score": 1.0}
            ],
        },
    )


class TestExtraFieldSections(unittest.TestCase):
    """Extra fields are accepted only in sections whose renderer fills them."""

    def test_listwise_prefix_and_suffix_accept_extra_fields(self):
        SingleTurnListwiseInferenceHandler(
            {
                **LISTWISE,
                "extra_fields": ["intent"],
                "prefix": "Query: {query}\nIntent: {intent}\n",
                "suffix": "Intent again: {intent}",
            }
        )

    def test_listwise_body_and_system_message_reject_extra_fields(self):
        for section, text in (
            ("body", "[{rank}] {candidate} {intent}\n"),
            ("system_message", "Mind the intent: {intent}"),
        ):
            with self.assertRaisesRegex(ValueError, f"'{section}' section") as ctx:
                SingleTurnListwiseInferenceHandler(
                    {**LISTWISE, "extra_fields": ["intent"], section: text}
                )
            self.assertIn("prefix, suffix", str(ctx.exception))

    def test_multiturn_and_pairwise_render_no_extra_fields(self):
        with self.assertRaisesRegex(ValueError, "no section of this template type"):
            MultiTurnListwiseInferenceHandler(
                {
                    "method": "multiturn_listwise",
                    "extra_fields": ["intent"],
                    "body_user": "[{rank}] {candidate}",
                    "prefix_user": "{query} {intent}",
                    "output_validation_regex": r"^\[\d+\]",
                    "output_extraction_regex": r"\[(\d+)\]",
                }
            )
        with self.assertRaisesRegex(ValueError, "no section of this template type"):
            PairwiseInferenceHandler(
                {
                    "method": "pairwise",
                    "extra_fields": ["intent"],
                    "body": "{query} {intent} {doc1} {doc2}",
                }
            )

    def test_pointwise_body_receives_extra_fields_on_every_path(self):
        handler = PointwiseInferenceHandler(
            {
                "method": "pointwise",
                "extra_fields": ["intent"],
                "body": "Query: {query} Intent: {intent} Document: {doc_content}",
            }
        )
        kwargs = dict(index=0, max_doc_tokens=100, tokenizer=FakeTokenizer())
        prompt = handler.generate_prompt(result({"intent": "find it"}), **kwargs)
        self.assertIn("Intent: find it", prompt)
        messages = handler.generate_chat_messages(
            result({"intent": "find it"}), **kwargs
        )
        self.assertIn("Intent: find it", messages[-1]["content"])
        # Few-shot examples belong to other queries: the field renders empty.
        fewshot = handler.generate_prompt(
            result({"intent": "find it"}),
            num_fewshot_examples=1,
            fewshot_examples=[
                {
                    "conversations": [
                        {"value": "Query: q Document: d"},
                        {"value": "true"},
                    ]
                }
            ],
            **kwargs,
        )
        self.assertIn("Intent: find it", fewshot)


class TestExtraFieldNames(unittest.TestCase):
    def test_names_of_built_in_placeholders_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\['query'\] are built-in"):
            PointwiseInferenceHandler(
                {
                    "method": "pointwise",
                    "message_roles": "reranker",
                    "instruction": "Judge relevance",
                    "extra_fields": ["query"],
                    "query_block": "#Query: {query}",
                }
            )
        with self.assertRaisesRegex(ValueError, r"\['num'\] are built-in"):
            SingleTurnListwiseInferenceHandler({**LISTWISE, "extra_fields": ["num"]})


class TestEmptyExtraFieldValues(unittest.TestCase):
    TEMPLATE = {
        "method": "pointwise",
        "message_roles": "reranker",
        "instruction": "Judge relevance",
        "extra_fields": ["reasoning"],
        "query_block": "#Reasoning Trace: {reasoning}\n#Query: {query}",
    }

    def query_role(self, extra_fields):
        messages = PointwiseInferenceHandler(
            dict(self.TEMPLATE)
        ).generate_chat_messages(
            result(extra_fields), index=0, max_doc_tokens=100, tokenizer=FakeTokenizer()
        )
        return {m["role"]: m["content"] for m in messages}["query"]

    def test_null_value_drops_its_line_like_an_absent_one(self):
        self.assertEqual(self.query_role({"reasoning": None}), "#Query: the query")
        self.assertEqual(self.query_role({}), "#Query: the query")
        self.assertNotIn("None", self.query_role({"reasoning": None}))


class TestMinOutputTokensKey(unittest.TestCase):
    def test_positive_integer_is_accepted_and_kept(self):
        handler = SingleTurnListwiseInferenceHandler(
            {**LISTWISE, "min_output_tokens": 1}
        )
        self.assertEqual(handler.template["min_output_tokens"], 1)

    def test_other_values_are_rejected(self):
        for bad in (0, -2, "1", True, 1.5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                SingleTurnListwiseInferenceHandler(
                    {**LISTWISE, "min_output_tokens": bad}
                )


if __name__ == "__main__":
    unittest.main()

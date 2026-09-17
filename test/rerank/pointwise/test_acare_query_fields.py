"""ACARe context-mode templates and the Query.fields plumbing they rely on."""

import unittest
from importlib.resources import files

import yaml
from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_inference_handler import (
    PointwiseInferenceHandler,
)
from test.rerank.conftest import FakeTokenizer

TEMPLATES = files("rank_llm.rerank.prompt_templates")

SUB_QUERY = "St John's Health Center celebrity death"
GLOBAL_QUERY = "What's the name of the hospital they died?"
REASONING = "Need the hospital where the Grammy-winning entertainer died."


def load(name):
    with (TEMPLATES / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_result(fields=None):
    return from_dict(
        data_class=Result,
        data={
            "query": {"text": SUB_QUERY, "qid": "q1", "fields": fields or {}},
            "candidates": [
                {"doc": {"contents": "A passage."}, "docid": "d1", "score": 0.5}
            ],
        },
    )


def query_role(template, fields):
    handler = PointwiseInferenceHandler(template)
    messages = handler.generate_chat_messages(
        make_result(fields), index=0, max_doc_tokens=100, tokenizer=FakeTokenizer()
    )
    return {m["role"]: m["content"] for m in messages}


class TestACAReQueryBlocks(unittest.TestCase):
    """Each mode must reproduce the query block from the model card verbatim."""

    def test_m1_reasoning_only(self):
        roles = query_role(load("acare_m1_template.yaml"), {"reasoning": REASONING})
        self.assertEqual(
            roles["query"], f"#Reasoning Trace: {REASONING}\n#Query: {SUB_QUERY}"
        )

    def test_m2_global_query_only(self):
        roles = query_role(load("acare_m2_template.yaml"), {"global_query": GLOBAL_QUERY})
        self.assertEqual(
            roles["query"],
            f"#Original Question (global query): {GLOBAL_QUERY}\n#Query: {SUB_QUERY}",
        )

    def test_m3_full_context(self):
        roles = query_role(
            load("acare_m3_template.yaml"),
            {"global_query": GLOBAL_QUERY, "reasoning": REASONING},
        )
        self.assertEqual(
            roles["query"],
            f"#Original Question (global query): {GLOBAL_QUERY}\n"
            f"#Reasoning Trace: {REASONING}\n"
            f"#Query: {SUB_QUERY}",
        )

    def test_instruction_is_sent_as_system_role(self):
        template = load("acare_m3_template.yaml")
        roles = query_role(template, {"global_query": GLOBAL_QUERY, "reasoning": REASONING})
        self.assertEqual(roles["system"], template["instruction"])

    def test_empty_field_drops_only_its_own_line(self):
        roles = query_role(
            load("acare_m3_template.yaml"),
            {"global_query": GLOBAL_QUERY, "reasoning": "   "},
        )
        self.assertEqual(
            roles["query"],
            f"#Original Question (global query): {GLOBAL_QUERY}\n#Query: {SUB_QUERY}",
        )

    def test_absent_field_behaves_like_empty(self):
        roles = query_role(load("acare_m3_template.yaml"), {"global_query": GLOBAL_QUERY})
        self.assertEqual(
            roles["query"],
            f"#Original Question (global query): {GLOBAL_QUERY}\n#Query: {SUB_QUERY}",
        )

    def test_stock_template_is_unchanged_by_the_feature(self):
        """A template without query_fields still sends the bare query text."""
        roles = query_role(load("pointwise_vllm_template.yaml"), {})
        self.assertEqual(roles["query"], SUB_QUERY)


class TestQueryFieldsValidation(unittest.TestCase):
    def test_undeclared_placeholder_names_itself(self):
        template = {
            "method": "pointwise",
            "message_roles": "reranker",
            "instruction": "Rank it",
            "query_block": "#Trace: {reasoning}\n#Query: {query}",
        }
        handler = PointwiseInferenceHandler(template)
        with self.assertRaises(KeyError) as ctx:
            handler.generate_chat_messages(
                make_result(), index=0, max_doc_tokens=100, tokenizer=FakeTokenizer()
            )
        self.assertIn("query_fields", str(ctx.exception))

    def test_query_fields_must_be_a_list_of_names(self):
        with self.assertRaises(ValueError):
            PointwiseInferenceHandler(
                {
                    "method": "pointwise",
                    "message_roles": "reranker",
                    "instruction": "Rank it",
                    "query_fields": "reasoning",
                }
            )


if __name__ == "__main__":
    unittest.main()

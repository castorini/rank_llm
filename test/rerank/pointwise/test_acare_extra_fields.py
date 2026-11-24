"""ACARe context-mode templates and the Query.extra_fields plumbing they rely on."""

import unittest
from importlib.resources import files
from unittest.mock import MagicMock, patch

import yaml
from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_inference_handler import (
    PointwiseInferenceHandler,
)
from rank_llm.rerank.pointwise.pointwise_vllm import PointwiseVLLM
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
            "query": {"text": SUB_QUERY, "qid": "q1", "extra_fields": fields or {}},
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
        roles = query_role(
            load("acare_m2_template.yaml"), {"global_query": GLOBAL_QUERY}
        )
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
        roles = query_role(
            template, {"global_query": GLOBAL_QUERY, "reasoning": REASONING}
        )
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
        roles = query_role(
            load("acare_m3_template.yaml"), {"global_query": GLOBAL_QUERY}
        )
        self.assertEqual(
            roles["query"],
            f"#Original Question (global query): {GLOBAL_QUERY}\n#Query: {SUB_QUERY}",
        )

    def test_stock_template_is_unchanged_by_the_feature(self):
        """A template without extra_fields still sends the bare query text."""
        roles = query_role(load("pointwise_vllm_template.yaml"), {})
        self.assertEqual(roles["query"], SUB_QUERY)


class _YesNoTokenizer:
    """`yes` and `no` are single tokens; anything else is several."""

    def encode(self, text, add_special_tokens=False, **kwargs):
        return {"yes": [1], "no": [2]}.get(text, [100, 101])

    def decode(self, ids, **kwargs):
        return {1: "yes", 2: "no"}.get(ids[0], "?")


class TestACAReAnswerTokens(unittest.TestCase):
    """Every ACARe mode scores like Qwen3-Reranker: P("yes") over P("yes") + P("no")."""

    MODES = (
        "acare_m1_template.yaml",
        "acare_m2_template.yaml",
        "acare_m3_template.yaml",
    )

    def test_each_mode_declares_lowercase_yes_no(self):
        for name in self.MODES:
            template = load(name)
            self.assertEqual(
                template["output_tokens"],
                {"positive": ["yes"], "negative": ["no"]},
                name,
            )
            PointwiseInferenceHandler(
                template
            )  # valid beside extra_fields / query_block

    def test_pointwise_vllm_takes_them_from_the_template(self):
        handler = MagicMock()
        handler.get_tokenizer.return_value = _YesNoTokenizer()
        for name in self.MODES:
            with patch(
                "rank_llm.rerank.pointwise.pointwise_vllm.VllmHandlerWithOpenAISDK",
                return_value=handler,
            ):
                reranker = PointwiseVLLM(
                    model="m",
                    base_url="http://127.0.0.1:8000/v1",
                    prompt_template_path=TEMPLATES / name,
                )
            self.assertEqual(reranker.answer_tokens.allowed_token_ids, [1, 2], name)


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
        self.assertIn("extra_fields", str(ctx.exception))

    def test_extra_fields_must_be_a_list_of_names(self):
        with self.assertRaises(ValueError):
            PointwiseInferenceHandler(
                {
                    "method": "pointwise",
                    "message_roles": "reranker",
                    "instruction": "Rank it",
                    "extra_fields": "reasoning",
                }
            )


if __name__ == "__main__":
    unittest.main()

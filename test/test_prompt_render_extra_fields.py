"""`rank-llm prompt render` with templates that use extra fields."""

import unittest

from rank_llm.cli.prompt_view import build_rendered_prompt_view

CANDIDATES = [{"docid": "d1", "text": "A passage."}]


def render(name, extra_fields=None):
    query = {"text": "the query", "qid": "q1"}
    if extra_fields is not None:
        query["extra_fields"] = extra_fields
    return build_rendered_prompt_view(name, {"query": query, "candidates": CANDIDATES})


class TestPromptRenderExtraFields(unittest.TestCase):
    def test_reranker_roles_template_renders_its_query_block(self):
        view = render(
            "acare_m3_template",
            {"global_query": "big question", "reasoning": "a trace"},
        )
        roles = [m["role"] for m in view["messages"]]
        self.assertEqual(roles, ["system", "query", "document"])
        self.assertEqual(
            view["messages"][1]["content"],
            "#Original Question (global query): big question\n#Reasoning Trace: a trace\n#Query: the query",
        )
        self.assertEqual(view["messages"][2]["content"], "A passage.")

    def test_absent_extra_fields_drop_their_lines(self):
        view = render("acare_m3_template")
        self.assertEqual(view["messages"][1]["content"], "#Query: the query")

    def test_listwise_prefix_receives_extra_fields(self):
        view = render("rubricranker_template", {"query_intent": "why it was asked"})
        self.assertIn("why it was asked", view["messages"][-1]["content"])

    def test_listwise_without_extra_fields_still_renders(self):
        view = render("rubricranker_rag_template")
        self.assertIn("the query", view["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()

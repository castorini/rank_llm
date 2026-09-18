"""RubricRanker template: upstream prompt fidelity and setwise-selection handling."""

import unittest
from importlib.resources import files

import yaml
from dacite import from_dict

from rank_llm.data import Result
from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM
from rank_llm.rerank.listwise.response_utils import clean_ranking_response
from rank_llm.rerank.listwise.singleturn_listwise_inference_handler import (
    SingleTurnListwiseInferenceHandler,
)

TEMPLATES = files("rank_llm.rerank.prompt_templates")

QUERY = "hospital where the entertainer died"
INTENT = "Looking for the Santa Monica hospital named in the obituary."
DOCS = ["First passage.", "Second passage.", "Third passage."]


def load_template(name="rubricranker_template.yaml"):
    with (TEMPLATES / name).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_result():
    return from_dict(
        data_class=Result,
        data={
            "query": {"text": QUERY, "qid": "q1", "fields": {"query_intent": INTENT}},
            "candidates": [
                {"doc": {"contents": doc}, "docid": f"d{i}", "score": 1.0 - i / 10}
                for i, doc in enumerate(DOCS)
            ],
        },
    )


def upstream_user_message(template):
    """The user message as RubricRanker's own client builds it.

    Mirrors vllm_setwise_rubricranker_reranker: documents numbered from 1 and
    joined by a blank line, substituted into the published user_message.
    """
    context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(DOCS))
    return (
        f"I will provide you with {len(DOCS)} documents, each indicated by a numerical "
        f"identifier []. Select and rank the documents based on their joint usefulness "
        f"for the search query: {QUERY}.\n\n"
        f"[QUERY INTENT]\n{INTENT}\n\n"
        f"Search Query: {QUERY}.\n\n"
        f"[DOCUMENTS]\n{context}\n\n"
        f"{template['suffix'].format(num=len(DOCS))}"
    )


class TestRubricRankerPrompt(unittest.TestCase):
    def setUp(self):
        self.template = load_template()
        self.handler = SingleTurnListwiseInferenceHandler(self.template)
        self.messages = self.handler.generate_prompt(
            make_result(), rank_start=0, rank_end=len(DOCS), max_length=100
        )

    def test_system_message_carries_the_rubrics(self):
        self.assertEqual(self.messages[0]["role"], "system")
        self.assertEqual(self.messages[0]["content"], self.template["system_message"])

    def test_user_message_matches_upstream_verbatim(self):
        self.assertEqual(
            self.messages[-1]["content"], upstream_user_message(self.template)
        )

    def test_query_intent_is_rendered(self):
        self.assertIn(f"[QUERY INTENT]\n{INTENT}", self.messages[-1]["content"])


class _Permuter:
    """Minimal carrier for ListwiseRankLLM.receive_permutation.

    The method needs only `_use_alpha` and `_remove_duplicate`; binding those
    exercises the real reordering logic without constructing a model
    coordinator. Note that receive_permutation swallows every exception and
    falls back to the input order, so a shim missing an attribute would make
    these assertions pass vacuously -- hence the explicit reordering check.
    """

    _use_alpha = False
    _remove_duplicate = ListwiseRankLLM._remove_duplicate
    receive_permutation = ListwiseRankLLM.receive_permutation


class TestRubricRankerRagVariant(unittest.TestCase):
    """The rag variant drops the intent block and frames the query as a question."""

    def setUp(self):
        self.template = load_template("rubricranker_rag_template.yaml")
        handler = SingleTurnListwiseInferenceHandler(self.template)
        self.messages = handler.generate_prompt(
            make_result(), rank_start=0, rank_end=len(DOCS), max_length=100
        )

    def test_user_message_matches_upstream_verbatim(self):
        context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(DOCS))
        expected = (
            f"I will provide you with {len(DOCS)} passages, each indicated by a numerical "
            f"identifier []. Select and rank the passages based on their joint usefulness "
            f"for the user question: {QUERY}.\n\n"
            f"[PASSAGES]\n{context}\n\n"
            f"{self.template['suffix'].format(num=len(DOCS), query=QUERY)}"
        )
        self.assertEqual(self.messages[-1]["content"], expected)

    def test_no_query_intent_block(self):
        self.assertNotIn("[QUERY INTENT]", self.messages[-1]["content"])

    def test_needs_no_query_fields(self):
        self.assertNotIn("query_fields", self.template)


class TestSetwiseSelectionParsing(unittest.TestCase):
    """RubricRanker emits a space-separated subset, not a full permutation."""

    def test_space_separated_selection_parses(self):
        # Every non-digit becomes a space, so "] [" widens the gap; the caller
        # splits on whitespace, which is what this guarantees.
        parsed = clean_ranking_response("[4] [2] [7] [11] [1]").split()
        self.assertEqual(parsed, ["4", "2", "7", "11", "1"])

    def test_partial_selection_keeps_unselected_order_behind(self):
        result = from_dict(
            data_class=Result,
            data={
                "query": {"text": QUERY, "qid": "q1"},
                "candidates": [
                    {"doc": {"contents": f"doc{i}"}, "docid": f"d{i}", "score": 0.0}
                    for i in range(5)
                ],
            },
        )
        reordered = _Permuter().receive_permutation(
            result, "[3] [1]", rank_start=0, rank_end=5
        )
        self.assertEqual(
            [c.docid for c in reordered.candidates],
            ["d2", "d0", "d1", "d3", "d4"],
        )


    def test_unparseable_output_falls_back_to_first_stage_order(self):
        result = from_dict(
            data_class=Result,
            data={
                "query": {"text": QUERY, "qid": "q1"},
                "candidates": [
                    {"doc": {"contents": f"doc{i}"}, "docid": f"d{i}", "score": 0.0}
                    for i in range(3)
                ],
            },
        )
        reordered = _Permuter().receive_permutation(
            result, "I could not select any documents.", rank_start=0, rank_end=3
        )
        self.assertEqual([c.docid for c in reordered.candidates], ["d0", "d1", "d2"])


if __name__ == "__main__":
    unittest.main()

"""A dict query's extra fields must survive normalization into Query.fields."""

import unittest

from rank_llm.cli.operations import normalize_direct_rerank_input

CANDIDATES = [{"docid": "d1", "score": 1.0, "text": "A passage."}]


def normalize(query):
    return normalize_direct_rerank_input({"query": query, "candidates": CANDIDATES})


class TestQueryFieldsPassthrough(unittest.TestCase):
    def test_explicit_fields_object_is_carried(self):
        out = normalize(
            {"text": "sub query", "qid": "q1", "fields": {"global_query": "big question"}}
        )
        self.assertEqual(out["query_text"], "sub query")
        self.assertEqual(out["query_id"], "q1")
        self.assertEqual(out["query_fields"], {"global_query": "big question"})

    def test_loose_extra_keys_are_collected_as_fields(self):
        out = normalize({"text": "sub query", "qid": "q1", "reasoning": "because"})
        self.assertEqual(out["query_fields"], {"reasoning": "because"})

    def test_explicit_fields_wins_over_loose_keys(self):
        out = normalize(
            {
                "text": "sub query",
                "qid": "q1",
                "reasoning": "ignored",
                "fields": {"reasoning": "used"},
            }
        )
        self.assertEqual(out["query_fields"], {"reasoning": "used"})

    def test_plain_string_query_has_no_fields(self):
        out = normalize("just a string")
        self.assertEqual(out["query_text"], "just a string")
        self.assertEqual(out["query_fields"], {})

    def test_bare_dict_query_has_no_fields(self):
        out = normalize({"text": "sub query", "qid": "q1"})
        self.assertEqual(out["query_fields"], {})

    def test_non_object_fields_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize({"text": "q", "qid": "q1", "fields": "not-an-object"})


if __name__ == "__main__":
    unittest.main()

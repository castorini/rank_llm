"""A dict query's `extra_fields` must survive normalization into Query.extra_fields."""

import json
import tempfile
import unittest
from pathlib import Path

from rank_llm.cli.operations import normalize_direct_rerank_input
from rank_llm.data import DataWriter, Query, Result

CANDIDATES = [{"docid": "d1", "score": 1.0, "text": "A passage."}]


def normalize(query):
    return normalize_direct_rerank_input({"query": query, "candidates": CANDIDATES})


class TestExtraFieldsPassthrough(unittest.TestCase):
    def test_extra_fields_object_is_carried(self):
        out = normalize(
            {
                "text": "sub query",
                "qid": "q1",
                "extra_fields": {"global_query": "big question"},
            }
        )
        self.assertEqual(out["query_text"], "sub query")
        self.assertEqual(out["query_id"], "q1")
        self.assertEqual(out["extra_fields"], {"global_query": "big question"})

    def test_other_keys_on_a_dict_query_are_ignored(self):
        out = normalize({"text": "sub query", "qid": "q1", "reasoning": "because"})
        self.assertEqual(out["extra_fields"], {})

    def test_plain_string_query_has_no_extra_fields(self):
        out = normalize("just a string")
        self.assertEqual(out["query_text"], "just a string")
        self.assertEqual(out["extra_fields"], {})

    def test_bare_dict_query_has_no_extra_fields(self):
        out = normalize({"text": "sub query", "qid": "q1"})
        self.assertEqual(out["extra_fields"], {})

    def test_non_object_extra_fields_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize({"text": "q", "qid": "q1", "extra_fields": "not-an-object"})


class TestExtraFieldsSerialisation(unittest.TestCase):
    def test_empty_extra_fields_are_omitted_and_set_ones_are_kept(self):
        plain = Result(query=Query(text="q", qid="1"))
        with_ctx = Result(
            query=Query(text="q", qid="2", extra_fields={"global_query": "g"})
        )
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "r.json"
            DataWriter([plain, with_ctx]).write_in_json_format(str(out))
            records = json.loads(out.read_text())
        self.assertEqual(records[0]["query"], {"text": "q", "qid": "1"})
        self.assertEqual(
            records[1]["query"],
            {"text": "q", "qid": "2", "extra_fields": {"global_query": "g"}},
        )


if __name__ == "__main__":
    unittest.main()

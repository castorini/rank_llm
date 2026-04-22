import unittest

from dacite import from_dict

from rank_llm.data import Request
from rank_llm.rerank.identity_reranker import IdentityReranker


class TestIdentityReranker(unittest.TestCase):
    def test_rerank_matches_first_of_batch(self):
        req = from_dict(
            data_class=Request,
            data={
                "query": {"text": "q", "qid": 1},
                "candidates": [
                    {"docid": "a", "score": 0.9, "doc": {"contents": "x"}},
                ],
            },
        )
        ir = IdentityReranker()
        one = ir.rerank(req, rank_end=1)
        many = ir.rerank_batch([req], rank_end=1)
        self.assertEqual(one.query.text, many[0].query.text)
        self.assertEqual(len(one.candidates), len(many[0].candidates))


if __name__ == "__main__":
    unittest.main()

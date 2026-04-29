import unittest

from rank_llm.rerank.pointwise.pointwise_vllm import _single_token_ids_from_strings


class TestSingleTokenIdsFromStrings(unittest.TestCase):
    def test_skips_multi_piece_encodings(self):
        class Tok:
            def encode(self, s: str, add_special_tokens: bool = False, **kwargs):
                if s == "yes":
                    return [1, 2]
                if s == "no":
                    return [3]
                return [99]

        ids = _single_token_ids_from_strings(Tok(), ("yes", "no"))
        self.assertEqual(ids, [3])


if __name__ == "__main__":
    unittest.main()

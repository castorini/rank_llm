import unittest

from dacite import from_dict
from transformers import T5Tokenizer

from rank_llm.data import Result
from rank_llm.rerank.pairwise.pairwise_inference_handler import PairwiseInferenceHandler

r = from_dict(
    data_class=Result,
    data={
        "query": {"text": "Sample Query", "qid": "q1"},
        "candidates": [
            {
                "doc": {
                    "contents": "Title1: Sample Title1 Content1: Sample Text1",
                },
                "docid": "d1",
                "score": 0.5,
            },
            {
                "doc": {
                    "contents": "Title2: Sample Title2 Content2: Sample Text2",
                },
                "docid": "d2",
                "score": 0.4,
            },
            {
                "doc": {
                    "contents": "Title3: Sample Title3 Content3: Sample Text3",
                },
                "docid": "d3",
                "score": 0.4,
            },
            {
                "doc": {
                    "contents": "Title4: Sample Title4 Content4: Sample Text4",
                },
                "docid": "d4",
                "score": 0.3,
            },
        ],
    },
)


VALID_PAIRWISE_TEMPLATE = {
    "method": "pairwise",
    "body": "Query: {query} Document0: {doc1} Document1: {doc2}",
}
INVALID_PAIRWISE_TEMPLATES = [
    {
        "method": "singleturn_listwise",
        "body": "{query} {doc1} {doc2}",
    },  # Wrong method type
    {
        "method": "pairwise",
        "body": "{query} {doc1}",
    },  # Missing required placeholder: {doc2}
    {
        "method": "pairwise",
        "body": "{query} {doc1} {doc2}",
        "unknown_key": "value",
    },  # Unknown key
]
tokenizer = T5Tokenizer.from_pretrained("castorini/duot5-3b-msmarco-10k")


class TestPairwiseInferenceHandler(unittest.TestCase):
    def test_pairwise_valid_template_initialization(self):
        pairwise_inference_handler = PairwiseInferenceHandler(VALID_PAIRWISE_TEMPLATE)
        self.assertEqual(pairwise_inference_handler.template, VALID_PAIRWISE_TEMPLATE)

    def test_invalid_templates(self):
        for template in INVALID_PAIRWISE_TEMPLATES:
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    PairwiseInferenceHandler(template)

    def test_body_generation(self):
        pairwise_inference_handler = PairwiseInferenceHandler(VALID_PAIRWISE_TEMPLATE)
        body_text_1 = pairwise_inference_handler._generate_body(
            result=r, index1=0, index2=1, single_doc_max_token=6000, tokenizer=tokenizer
        )
        body_text_2 = pairwise_inference_handler._generate_body(
            result=r, index1=0, index2=2, single_doc_max_token=6000, tokenizer=tokenizer
        )
        expected_body_1 = "Query: Sample Query Document0: Title1: Sample Title1 Content1: Sample Text1 Document1: Title2: Sample Title2 Content2: Sample Text2"
        expected_body_2 = "Query: Sample Query Document0: Title1: Sample Title1 Content1: Sample Text1 Document1: Title3: Sample Title3 Content3: Sample Text3"

        self.assertEqual(body_text_1, expected_body_1)
        self.assertEqual(body_text_2, expected_body_2)

    def test_prompt_generation(self):
        pairwise_inference_handler = PairwiseInferenceHandler(VALID_PAIRWISE_TEMPLATE)
        prompt_text_1 = pairwise_inference_handler.generate_prompt(
            result=r, index1=0, index2=1, max_token=6000, tokenizer=tokenizer
        )
        prompt_text_2 = pairwise_inference_handler.generate_prompt(
            result=r, index1=0, index2=2, max_token=6000, tokenizer=tokenizer
        )
        expected_prompt_1 = "Query: Sample Query Document0: Title1: Sample Title1 Content1: Sample Text1 Document1: Title2: Sample Title2 Content2: Sample Text2"
        expected_prompt_2 = "Query: Sample Query Document0: Title1: Sample Title1 Content1: Sample Text1 Document1: Title3: Sample Title3 Content3: Sample Text3"

        self.assertEqual(prompt_text_1, expected_prompt_1)
        self.assertEqual(prompt_text_2, expected_prompt_2)


if __name__ == "__main__":
    unittest.main()

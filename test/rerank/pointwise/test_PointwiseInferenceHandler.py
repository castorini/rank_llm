import unittest

from dacite import from_dict
from transformers import T5Tokenizer

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_inference_handler import (
    PointwiseInferenceHandler,
)

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


VALID_POINTWISE_TEMPLATE = {
    "method": "pointwise",
    "body": "Query: {query} Document: {doc_content} Relevance: ",
}
INVALID_POINTWISE_TEMPLATES = [
    {
        "method": "pairwise",
        "body": "{query} {doc_content}",
    },  # Wrong method type
    {
        "method": "pointwise",
        "body": "{doc_content}",
    },  # Missing required placeholder: {query}
    {
        "method": "pointwise",
        "body": "{query} {doc_content}",
        "unknown_key": "value",
    },  # Unknown key
]
tokenizer = T5Tokenizer.from_pretrained("castorini/monot5-3b-msmarco-10k")


class TestPointwiseInferenceHandler(unittest.TestCase):
    def test_pointwise_valid_template_initialization(self):
        inference_handler = PointwiseInferenceHandler(VALID_POINTWISE_TEMPLATE)
        self.assertEqual(inference_handler.template, VALID_POINTWISE_TEMPLATE)

    def test_invalid_templates(self):
        for template in INVALID_POINTWISE_TEMPLATES:
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    PointwiseInferenceHandler(template)

    def test_body_generation(self):
        inference_handler = PointwiseInferenceHandler(VALID_POINTWISE_TEMPLATE)
        body_text_1 = inference_handler._generate_body(
            result=r, index=0, max_doc_tokens=6000, tokenizer=tokenizer
        )
        body_text_2 = inference_handler._generate_body(
            result=r, index=1, max_doc_tokens=6000, tokenizer=tokenizer
        )
        expected_body_1 = "Query: Sample Query Document: Title1: Sample Title1 Content1: Sample Text1 Relevance: "
        expected_body_2 = "Query: Sample Query Document: Title2: Sample Title2 Content2: Sample Text2 Relevance: "

        self.assertEqual(body_text_1, expected_body_1)
        self.assertEqual(body_text_2, expected_body_2)

    def test_prompt_generation(self):
        inference_handler = PointwiseInferenceHandler(VALID_POINTWISE_TEMPLATE)
        prompt_text_1 = inference_handler.generate_prompt(
            result=r, index=0, max_doc_tokens=6000, tokenizer=tokenizer
        )
        prompt_text_2 = inference_handler.generate_prompt(
            result=r, index=1, max_doc_tokens=6000, tokenizer=tokenizer
        )
        expected_prompt_1 = "Query: Sample Query Document: Title1: Sample Title1 Content1: Sample Text1 Relevance: "
        expected_prompt_2 = "Query: Sample Query Document: Title2: Sample Title2 Content2: Sample Text2 Relevance: "

        self.assertEqual(prompt_text_1, expected_prompt_1)
        self.assertEqual(prompt_text_2, expected_prompt_2)


if __name__ == "__main__":
    unittest.main()

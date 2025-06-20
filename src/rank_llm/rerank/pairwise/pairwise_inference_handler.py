from typing import Any, Dict

from transformers import T5Tokenizer

from rank_llm.data import Result
from rank_llm.rerank.inference_handler import BaseInferenceHandler


class PairwiseInferenceHandler(BaseInferenceHandler):
    def __init__(self, template: Dict[str, str]):
        super().__init__(template)

    def _validate_template(self, template: Dict[str, str], strict: bool = False):
        TEMPLATE_SECTIONS = {
            # Format:
            # "template_key": {
            #    "required": True/False,  # Whether the section itself is mandatory
            #    "required_placeholders": set(),  # Placeholders that must exist in this section
            #    "allowed_placeholders": set()    # All allowed placeholders (including required ones)
            # }
            "body": {
                "required": True,
                "required_placeholders": {"query", "doc1", "doc2"},
                "allowed_placeholders": set(),
            },
        }

        # Validate the method value
        if template["method"] != "pairwise":
            raise ValueError(
                f'Incorrect method type, expected "pairwise", got {template["method"]}'
            )

        self._general_validation(
            template=template, template_section=TEMPLATE_SECTIONS, strict=strict
        )

    # TODO (issue #273): May need to add prefix/suffix generation function later

    def _generate_body(
        self,
        result: Result,
        index1: int,
        index2: int,
        single_doc_max_token: int,
        tokenizer: T5Tokenizer,
    ) -> str:
        doc1_raw = self._convert_doc_to_prompt_content(
            result.candidates[index1].doc, max_length=single_doc_max_token
        )
        doc2_raw = self._convert_doc_to_prompt_content(
            result.candidates[index2].doc, max_length=single_doc_max_token
        )

        doc1_tokens = tokenizer.encode(
            doc1_raw, truncation=True, max_length=single_doc_max_token
        )
        doc2_tokens = tokenizer.encode(
            doc2_raw, truncation=True, max_length=single_doc_max_token
        )

        query = self._replace_number(result.query.text)
        doc1 = tokenizer.decode(doc1_tokens, skip_special_tokens=True)
        doc2 = tokenizer.decode(doc2_tokens, skip_special_tokens=True)

        fmt_values = {"query": query, "doc1": doc1, "doc2": doc2}
        body_text = self._format_template(template_key="body", fmt_values=fmt_values)

        return body_text

    def generate_prompt(self, result: Result, **kwargs: Any) -> str:
        try:
            index1 = kwargs["index1"]
            index2 = kwargs["index2"]
            max_token = kwargs["max_token"]
            tokenizer = kwargs["tokenizer"]
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        single_doc_max_token = max_token // 2

        prompt = self._generate_body(
            result=result,
            index1=index1,
            index2=index2,
            single_doc_max_token=single_doc_max_token,
            tokenizer=tokenizer,
        )
        return prompt.replace("<unk>", "")

from string import Formatter
from typing import Any, Dict

from transformers import T5Tokenizer

from rank_llm.data import Result
from rank_llm.rerank.inference_handler import BaseInferenceHandler


class PointwiseInferenceHandler(BaseInferenceHandler):
    def __init__(self, template: Dict[str, str]):
        self._formatter = Formatter()
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
                "required_placeholders": {"query", "doc_content"},
                "allowed_placeholders": set(),
            },
        }

        # Validate the method value
        if template["method"] != "pointwise":
            raise ValueError(
                f'Incorrect method type, expected "pointwise", got {template["method"]}'
            )

        self._general_validation(template_section=TEMPLATE_SECTIONS, strict=strict)

    # TODO (issue #273): May need to add prefix/suffix generation function later

    def _generate_body(
        self, result: Result, index: int, max_doc_tokens: int, tokenizer: T5Tokenizer
    ) -> str:
        query = self._replace_number(result.query.text)
        doc_raw = self._convert_doc_to_prompt_content(
            result.candidates[index].doc, max_length=max_doc_tokens
        )
        doc_tokens = tokenizer.encode(
            doc_raw, truncation=True, max_length=max_doc_tokens
        )
        doc = tokenizer.decode(doc_tokens, skip_special_tokens=True)

        fmt_values = {"query": query, "doc_content": doc}
        body_text = self._format_template(template_key="body", fmt_values=fmt_values)

        return body_text

    def generate_prompt(self, result: Result, **kwargs: Any) -> str:
        try:
            index = kwargs["index"]
            max_doc_tokens = kwargs["max_doc_tokens"]
            tokenizer = kwargs["tokenizer"]
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        prompt = self._generate_body(
            result=result,
            index=index,
            max_doc_tokens=max_doc_tokens,
            tokenizer=tokenizer,
        )
        return prompt.replace("<unk>", "")

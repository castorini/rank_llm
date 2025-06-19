from string import Formatter
from typing import Any, Dict

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
                f'Incorrect method type, expected "pairwise", got {template["method"]}'
            )

        # Validate the required template keys
        missing_template_keys = [
            key
            for key, config in TEMPLATE_SECTIONS.items()
            if key not in template and config["required"]
        ]
        if missing_template_keys:
            raise ValueError(f"Missing required template keys: {missing_template_keys}")

        # Validate the rest of the template keys
        for template_key, template_text in template.items():
            if template_key == "method":
                continue
            if template_key not in TEMPLATE_SECTIONS:
                raise ValueError(f"Unsupported template section: {template_key}")

            section = TEMPLATE_SECTIONS[template_key]
            required_placeholders = section["required_placeholders"]
            allowed_placeholders = (
                required_placeholders | section["allowed_placeholders"]
            )
            used_placeholders = {
                name
                for _, name, _, _ in self._formatter.parse(template_text)
                if name is not None
            }
            missing_placeholders = required_placeholders - used_placeholders
            if missing_placeholders:
                raise ValueError(
                    f"Missing placeholders in {template_key} section: {missing_placeholders}"
                )

            unsupported_placeholders = used_placeholders - allowed_placeholders
            if unsupported_placeholders:
                msg = f"Unsupported placeholders in {template_key} section: {unsupported_placeholders}"
                if strict:
                    raise ValueError(msg)
                else:
                    print(msg)

    # TODO (issue #273): May need to add prefix/suffix generation function later

    def _generate_body(
        self,
        result: Result,
        index: int,
        max_doc_tokens: int,
    ) -> str:
        doc_content = self._convert_doc_to_prompt_content(
            result.candidates[index].doc, max_length=max_doc_tokens
        )

        query = result.query.text

        fmt_values = {"query": query, "doc_content": doc_content}
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

        prompt = self._generate_body(
            result=result,
            index1=index1,
            index2=index2,
            max_token=max_token,
            tokenizer=tokenizer,
        )
        return prompt.replace("<unk>", "")

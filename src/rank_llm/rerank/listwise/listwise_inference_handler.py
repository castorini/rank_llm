from typing import Any, Dict, List, Optional, Tuple

from rank_llm.data import Result
from rank_llm.rerank.inference_handler import BaseInferenceHandler


# TODO(issue #237): Need to modify functions for this class
class ListwiseInferenceHandler(BaseInferenceHandler):
    def __init__(self, template: Dict[str, str]):
        super().__init__(template)

        self._validate_template(self.template)

    def _validate_template(self, template: Dict[str, str]):
        required_template_keys = {
            "body": ["{rank}", "{candidate}"],
        }

        allowed_template_keys = {
            "system_message": [],
            "prefix": ["{num}", "{query}"],
            "suffix": ["{num}", "{query}"],
        }

        # Validate the method value
        if template["method"] != "listwise":
            raise ValueError(
                f"Incorrect method type, expected \"listwise\", got {template['method']}"
            )

        # Validate the required keys
        missing_template_keys = [
            key for key in required_template_keys if key not in template
        ]
        if missing_template_keys:
            raise ValueError(f"Missing required template keys: {missing_template_keys}")

        # Validate the rest of the template keys
        for template_key, template_value in template.items():
            if (
                template_key not in required_template_keys
                and template_key not in allowed_template_keys
            ):
                raise ValueError(f"Unknown template key: {template_key}")
            if template_key in required_template_keys:
                for keyword in required_template_keys[template_key]:
                    if keyword not in template_value:
                        raise ValueError(
                            f"Missing required keyword {keyword} in {template_key} section"
                        )
            if template_key in allowed_template_keys:
                for keyword in allowed_template_keys[template_key]:
                    if keyword not in template_value:
                        raise ValueError(
                            f"Missing required keywords: {keyword} in {template_key} section"
                        )

        print("Template validated successfully!")

    def _generate_prefix(
        self, num: Optional[int] = None, query: Optional[str] = None
    ) -> str:
        pass

    def _generate_suffix(
        self, num: Optional[int] = None, query: Optional[int] = None
    ) -> str:
        pass

    def _generate_body(self, result: Result) -> str:
        pass

    def generate_prompt(self, result: Result, **kwargs: Any) -> Tuple[str, int]:
        try:
            rank_start = kwargs["rank_start"]
            rank_end = kwargs["rank_end"]
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")
        pass

    def generate_prompt_batched(
        self,
        result: Result,
        **kwargs: Any,
    ) -> List[Tuple[str, int]]:
        try:
            rank_start = kwargs["rank_start"]
            rank_end = kwargs["rank_end"]
            batch_size = kwargs["batch_size"]
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")
        pass

    def _clean_response(self, response: str, **kwargs: Any) -> str:
        ALPH_START_IDX = ord("A") - 1
        use_alpha = kwargs.get("use_alpha", False)

        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        fake_numbers_map = str.maketrans(
            "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉①②③④⑤⑥⑦⑧⑨❶❷❸❹❺❻❼❽❾０１２３４５６７８９🄀🄁🄂🄃🄄🄅🄆🄇🄈🄉",
            "0123456789012345678912345678912345678901234567890123456789",
        )
        response = response.translate(fake_numbers_map)

        new_response = ""
        if use_alpha:
            for c in response:
                if not c.isalpha():
                    new_response += " "
                else:
                    new_response += str(ord(c) - ALPH_START_IDX)
            new_response = new_response.strip()
        else:
            for c in response:
                if not c.isdigit():
                    new_response += " "
                else:
                    new_response += c
            new_response = new_response.strip()

        return new_response

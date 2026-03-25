from abc import ABC, abstractmethod
from typing import Any

from rank_llm.data import Result
from rank_llm.rerank.inference_handler import BaseInferenceHandler


class ListwiseInferenceHandler(BaseInferenceHandler, ABC):
    ALPH_START_IDX = ord("A") - 1

    def __init__(self, template: dict[str, str]):
        super().__init__(template)

    @abstractmethod
    def _generate_prefix_suffix(
        self, num: int, query: str, **kwargs: Any
    ) -> tuple[str | list[dict[str, str]], str]:
        pass

    @abstractmethod
    def _generate_body(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        max_length: int,
        use_alpha: bool,
    ) -> str | list[dict[str, str]]:
        pass

    def _generate_fewshot_prompt(
        self,
        num_examples: int = 0,
        examples: list[dict[str, list[dict[str, str]]]] | None = None,
    ) -> list[dict[str, str]]:
        if examples is None:
            examples = []
        few_shot_prompt = []
        for ex in examples[: min(num_examples, len(examples))]:
            for turn in ex["conversations"]:
                few_shot_prompt.append({"role": turn["role"], "content": turn["value"]})
        return few_shot_prompt

    def _clean_response(self, response: str, **kwargs: Any) -> str:
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
                    new_response += str(ord(c) - self.ALPH_START_IDX)
            new_response = new_response.strip()
        else:
            for c in response:
                if not c.isdigit():
                    new_response += " "
                else:
                    new_response += c
            new_response = new_response.strip()

        return new_response

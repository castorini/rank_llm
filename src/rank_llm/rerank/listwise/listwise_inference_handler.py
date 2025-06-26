from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from rank_llm.data import Result
from rank_llm.rerank.inference_handler import BaseInferenceHandler


class ListwiseInferenceHandler(BaseInferenceHandler, ABC):
    ALPH_START_IDX = ord("A") - 1

    def __init__(self, template: Dict[str, str]):
        super().__init__(template)

    @abstractmethod
    def _generate_prefix_suffix(
        self, num: int, query: str, **kwargs: Any
    ) -> Tuple[str | List[Dict[str, str]], str]:
        pass

    @abstractmethod
    def _generate_body(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        max_length: int,
        use_alpha: bool,
    ) -> str | List[Dict[str, str]]:
        pass

    def _generate_fewshot_prompt(
        self,
        num_examples: int = 0,
        examples: List[Dict[str, List[Dict[str, str]]]] = [],
        **kwargs: Any,
    ) -> List[Dict[str, str]] | str:
        is_messages = kwargs.get("is_messages", True)
        if is_messages:
            few_shot_prompt = []
            for ex in examples[: min(num_examples, len(examples))]:
                for turn in ex["conversations"]:
                    few_shot_prompt.append(
                        {"role": turn["role"], "content": turn["value"]}
                    )
            return few_shot_prompt
        else:  # string format
            example_messages = []
            for ex in examples[: min(num_examples, len(examples))]:
                if "conversations" in ex and len(ex) >= 2:
                    example_messages.append(
                        f"Example Input:\n{ex['conversations'][0]['value'].strip()}\n"
                        f"Expected Response:\n{ex['conversations'][1]['value'].strip()}"
                    )
            example_text = "\n\n".join(example_messages)

            if "few_shot" in self.template:
                fmt_values = {"examples": example_text}
                return self._format_template(
                    template_key="few_shot", fmt_values=fmt_values
                )
            else:
                return f"In response to the query, rank the passages. Ignore aspects like length, complexity, or writing style, and concentrate on passages that provide a comprehensive understanding of the query. Take into account any inaccuracies or vagueness in the passages when determining their relevance.\nExamples:\n{example_text}"

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

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

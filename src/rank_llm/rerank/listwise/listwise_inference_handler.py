from typing import Any, Dict, List, Optional, Tuple

from rank_llm.data import Result
from rank_llm.rerank.inference_handler import BaseInferenceHandler


class ListwiseInferenceHandler(BaseInferenceHandler):
    def __init__(self, template: Dict[str, str]):
        super().__init__(template)

    def _validate_template(self, template: Dict[str, str]):
        pass

    def _replace_key(self, template_part: str, **kwargs: Any) -> str:
        pass

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

    def generate_prompt(
        self, result: Result, rank_start: int, rank_end: int, batch_size: int
    ) -> Tuple[str, int] | List[Tuple[Dict[str, str], int]]:
        pass

    def response_analyzer(self, response: str):
        pass

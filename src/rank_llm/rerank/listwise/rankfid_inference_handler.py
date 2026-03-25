from typing import Any

from rank_llm.data import Result, TemplateSectionConfig
from rank_llm.rerank.inference_handler import BaseInferenceHandler


class RankFIDInferenceHandler(BaseInferenceHandler):
    def __init__(self, template: dict[str, str]):
        super().__init__(template)

    def _validate_template(self, template: dict[str, str], strict: bool = False):
        TEMPLATE_SECTIONS: dict[str, TemplateSectionConfig] = {
            "method": TemplateSectionConfig(
                required=True,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
            "text": TemplateSectionConfig(
                required=True,
                required_placeholders={"query", "passage"},
                allowed_placeholders={"index"},
            ),
            "query": TemplateSectionConfig(
                required=False,
                required_placeholders={"query"},
                allowed_placeholders=set(),
            ),
            "output_validation_regex": TemplateSectionConfig(
                required=True,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
            "output_extraction_regex": TemplateSectionConfig(
                required=True,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
        }

        # Validate the method value
        if template["method"] != "rankfid":
            raise ValueError(
                f'Incorrect method type, expected "rankfid", got {template["method"]}'
            )

        self._general_validation(
            template=template,
            template_section=TEMPLATE_SECTIONS,
            strict=strict,
            check_query=True,
        )

    def _generate_query(self, query: str) -> str:
        query_fmt_values = {"query": query}

        query_string = self._format_template(
            template_key="query", fmt_values=query_fmt_values
        )

        return query_string

    def _generate_fewshot_prompt(
        self,
        num_examples: int = 0,
        examples: list[dict[str, list[dict[str, str]]]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        raise NotImplementedError("Fewshot examples are not supported for RankFiD!")

    def generate_prompt(self, result: Result, **kwargs: Any) -> list[dict[str, str]]:
        try:
            rank_start = kwargs["rank_start"]
            rank_end = kwargs["rank_end"]
            max_tokens = kwargs["max_tokens"]
        except KeyError as err:
            raise ValueError(f"Missing required parameter: {err}") from err

        prompts = []
        query = result.query.text
        query_string = self._generate_query(query=query)

        rank = 0
        for cand in result.candidates[rank_start:rank_end]:
            rank += 1
            passage = self._convert_doc_to_prompt_content(cand.doc, max_tokens)

            fmt_values = {"query": query, "passage": passage, "index": rank}
            single_text = self._format_template(
                template_key="text", fmt_values=fmt_values
            )

            if self.template.get("query"):
                prompts.append({"query": query_string, "text": single_text})
            else:
                prompts.append({"text": single_text})

        return prompts

from typing import Any, Dict, List, Tuple

from rank_llm.data import Result, TemplateSectionConfig
from rank_llm.rerank.listwise.listwise_inference_handler import ListwiseInferenceHandler


class SingleTurnListwiseInferenceHandler(ListwiseInferenceHandler):
    def __init__(self, template: Dict[str, str]):
        super().__init__(template)

    def _validate_template(self, template: Dict[str, str], strict: bool = False):
        TEMPLATE_SECTIONS: Dict[str, TemplateSectionConfig] = {
            "body": TemplateSectionConfig(
                required=True,
                required_placeholders={"rank", "candidate"},
                allowed_placeholders=set(),
            ),
            "system_message": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
            "prefix": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders={"query", "num"},
            ),
            "suffix": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders={"query", "num", "psg_ids"},
            ),
        }

        # Validate the method value
        if template["method"] != "singleturn_listwise":
            raise ValueError(
                f'Incorrect method type, expected "singleturn_listwise", got {template["method"]}'
            )

        self._general_validation(
            template=template,
            template_section=TEMPLATE_SECTIONS,
            strict=strict,
            check_query=True,
        )

    def _generate_prefix_suffix(
        self, num: int, query: str, **kwargs: Any
    ) -> Tuple[str, str]:
        suffix_placeholders = [
            name
            for _, name, _, _ in self._formatter.parse(self.template["suffix"])
            if name is not None
        ]

        prefix_fmt_values = suffix_fmt_values = {"num": num, "query": query}

        if "psg_ids" in suffix_placeholders:  # Used in RankLRL prompt mode
            rank_start = kwargs["rank_start"]
            rank_end = kwargs["rank_end"]
            psg_ids = []
            for rank in range(rank_end - rank_start):
                psg_ids.append(f"PASSAGE{rank+1}")
            psg_ids_str = "[" + ", ".join(psg_ids) + "]"
            suffix_fmt_values["psg_ids"] = psg_ids_str

        prefix_text = self._format_template(
            template_key="prefix", fmt_values=prefix_fmt_values
        )
        suffix_text = self._format_template(
            template_key="suffix", fmt_values=suffix_fmt_values
        )

        return prefix_text, suffix_text

    def _generate_body(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        max_length: int,
        use_alpha: bool,
    ) -> str:
        body_text = ""
        rank = 0
        for cand in result.candidates[rank_start:rank_end]:
            rank += 1

            content = self._convert_doc_to_prompt_content(cand.doc, max_length)
            identifier = chr(self.ALPH_START_IDX + rank) if use_alpha else str(rank)

            fmt_values = {"rank": identifier, "candidate": content}
            single_text = self._format_template(
                template_key="body", fmt_values=fmt_values
            )

            body_text += single_text

        return body_text

    def generate_prompt(self, result: Result, **kwargs: Any) -> List[Dict[str, str]]:
        try:
            rank_start = kwargs["rank_start"]
            rank_end = kwargs["rank_end"]
            max_length = kwargs["max_length"]
            use_alpha = kwargs.get("use_alpha", False)
            num_fewshot_examples = kwargs.get("num_fewshot_examples", 0)
            fewshot_examples = kwargs.get("fewshot_examples", [])
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        query = result.query.text
        query = self._replace_number(query)
        num = len(result.candidates[rank_start:rank_end])

        prompt_messages = [
            {"role": "system", "content": system_message}
            for system_message in [self.template.get("system_message", "")]
            if system_message
        ]

        if num_fewshot_examples > 0 and fewshot_examples:
            examples = self._generate_fewshot_prompt(
                num_examples=num_fewshot_examples,
                examples=fewshot_examples,
            )
            prompt_messages.extend(examples)

        prefix_text, suffix_text = self._generate_prefix_suffix(
            num=num, query=query, rank_start=rank_start, rank_end=rank_end
        )
        body_text = self._generate_body(
            result=result,
            rank_start=rank_start,
            rank_end=rank_end,
            max_length=max_length,
            use_alpha=use_alpha,
        )
        prompt_text = ""

        if prefix_text:
            prompt_text += prefix_text
        prompt_text += body_text
        if suffix_text:
            prompt_text += suffix_text

        prompt_messages.append({"role": "user", "content": prompt_text})

        return prompt_messages

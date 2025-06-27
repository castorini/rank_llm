from typing import Any, Dict, List, Tuple

from rank_llm.data import Result, TemplateSectionConfig
from rank_llm.rerank.listwise.listwise_inference_handler import ListwiseInferenceHandler


class MultiTurnListwiseInferenceHandler(ListwiseInferenceHandler):
    def __init__(self, template: Dict[str, str]):
        super().__init__(template)

    def _validate_template(self, template: Dict[str, str], strict: bool = False):
        TEMPLATE_SECTIONS: Dict[str, TemplateSectionConfig] = {
            "body_user": TemplateSectionConfig(
                required=True,
                required_placeholders={"rank", "candidate"},
                allowed_placeholders=set(),
            ),
            "system_message": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
            "prefix_assistant": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders={"query", "num"},
            ),
            "body_assistant": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders={"rank"},
            ),
            "prefix_user": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders={"query", "num"},
            ),
            "suffix_user": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders={"query", "num"},
            ),
        }

        # Validate the method value
        if template["method"] != "multiturn_listwise":
            raise ValueError(
                f'Incorrect method type, expected "multiturn_listwise", got {template["method"]}'
            )

        self._general_validation(
            template=template,
            template_section=TEMPLATE_SECTIONS,
            strict=strict,
            check_query=True,
        )

        # Validate if assistant section is present
        if "prefix_assistant" not in template and "body_assistant" not in template:
            raise ValueError(
                "One of prefix_assistant and body_assistant sections must be present if the template method is multiturn_listwise"
            )
        if "prefix_assistant" in template and "prefix_user" not in template:
            raise ValueError(
                "prefix_user section must be present if prefix_assistant section is used in the template"
            )
        if (
            "body_assistant" in template
            and "prefix_user" in template
            and "prefix_assistant" not in template
        ):
            raise ValueError(
                "prefix_assistant section must be present if body_assisstant and prefix_user sections are used in the template"
            )

    def _generate_prefix_suffix(
        self, num: int, query: str, **kwargs: Any
    ) -> Tuple[str | List[Dict[str, str]], str]:
        prefix_fmt_values = suffix_fmt_values = {"num": num, "query": query}

        prefix_text = self._format_template(
            template_key="prefix_user", fmt_values=prefix_fmt_values
        )
        suffix_text = self._format_template(
            template_key="suffix_user", fmt_values=suffix_fmt_values
        )

        if not prefix_text:
            return "", suffix_text

        assistant_text = self._format_template(
            template_key="prefix_assistant", fmt_values=prefix_fmt_values
        )
        return [
            {"role": "user", "content": prefix_text},
            {"role": "assistant", "content": assistant_text},
        ], suffix_text

    def _generate_body(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        max_length: int,
        use_alpha: bool = False,
        is_conversational: bool = False,
    ) -> str | List[Dict[str, str]]:
        if is_conversational:
            body_prompt = []
        else:
            body_prompt = ""

        rank = 0
        for cand in result.candidates[rank_start:rank_end]:
            rank += 1
            content = self._convert_doc_to_prompt_content(cand.doc, max_length)
            content = self._replace_number(content)
            identifier = chr(self.ALPH_START_IDX + rank) if use_alpha else str(rank)
            body_fmt_values = {"rank": identifier, "candidate": content}
            body_text = self._format_template("body_user", body_fmt_values)

            if is_conversational:
                assistant_fmt_values = {"rank": identifier}
                assistant_text = self._format_template(
                    "body_assistant", assistant_fmt_values
                )
                body_prompt.extend(
                    [
                        {
                            "role": "user",
                            "content": body_text,
                        },
                        {"role": "assistant", "content": assistant_text},
                    ]
                )
            else:
                body_prompt += body_text

        return body_prompt

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
            fewshot_prompt = self._generate_fewshot_prompt(
                num_examples=num_fewshot_examples,
                examples=fewshot_examples,
            )
            prompt_messages.extend(fewshot_prompt)

        prefix_prompt, suffix_text = self._generate_prefix_suffix(num=num, query=query)
        is_conversational_body = "body_assistant" in self.template
        body_prompt = self._generate_body(
            result=result,
            rank_start=rank_start,
            rank_end=rank_end,
            max_length=max_length,
            use_alpha=use_alpha,
            is_conversational=is_conversational_body,
        )

        if prefix_prompt and isinstance(prefix_prompt, list):
            prompt_messages.extend(prefix_prompt)
        if is_conversational_body and isinstance(body_prompt, list):
            prompt_messages.extend(body_prompt)
            if suffix_text:
                prompt_messages.append({"role": "user", "content": suffix_text})
        else:
            prompt_text = body_prompt
            if suffix_text:
                prompt_text += suffix_text
            prompt_messages.append({"role": "user", "content": prompt_text})

        return prompt_messages

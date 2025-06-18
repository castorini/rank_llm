from typing import Any, Dict, List, Tuple

from rank_llm.data import Result
from rank_llm.rerank.listwise.listwise_inference_handler import ListwiseInferenceHandler


class SingleTurnListwiseInferenceHandler(ListwiseInferenceHandler):
    def __init__(self, template: Dict[str, str]):
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
                "required_placeholders": {"rank", "candidate"},
                "allowed_placeholders": set(),
            },
            "system_message": {
                "required": False,
                "required_placeholders": set(),
                "allowed_placeholders": set(),
            },
            "prefix": {
                "required": False,
                "required_placeholders": set(),
                "allowed_placeholders": {"query", "num"},
            },
            "suffix": {
                "required": False,
                "required_placeholders": set(),
                "allowed_placeholders": {"query", "num", "psg_ids"},
            },
        }

        # Validate the method value
        if template["method"] != "singleturn_listwise":
            raise ValueError(
                f'Incorrect method type, expected "listwise_norm", got {template["method"]}'
            )

        # Validate the required template keys
        missing_template_keys = [
            key
            for key, config in TEMPLATE_SECTIONS.items()
            if key not in template and config["required"]
        ]
        if missing_template_keys:
            raise ValueError(f"Missing required template keys: {missing_template_keys}")

        query_present = False if "prefix" in template or "suffix" in template else True

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

            if "query" in used_placeholders:
                query_present = True

        if not query_present:
            raise ValueError(
                "query placeholder must be present in prefix and/or suffix"
            )

        print("Template validated successfully!")

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
        prefix_text, suffix_text = self._generate_prefix_suffix(num, query)
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

import re
from string import Formatter
from typing import Any, Dict, List, Set, Tuple

from ftfy import fix_text

from rank_llm.data import Result
from rank_llm.rerank.inference_handler import BaseInferenceHandler

ALPH_START_IDX = ord("A") - 1


class ListwiseInferenceHandler(BaseInferenceHandler):
    def __init__(self, template: Dict[str, str]):
        super().__init__(template)

        self._validate_template(self.template)

    def _validate_template(self, template: Dict[str, str], strict: bool = False):
        required_template_keys = {
            "body": {"required": ["rank", "candidate"], "allowed": []},
        }

        allowed_template_keys = {
            "system_message": {"required": [], "allowed": []},
            "prefix": {"required": ["num", "query"], "allowed": []},
            "suffix": {"required": ["num", "query"], "allowed": ["psg_ids"]},
        }

        formatter = Formatter()

        # Validate the method value
        if template["method"] != "listwise":
            raise ValueError(
                f'Incorrect method type, expected "listwise", got {template["method"]}'
            )

        # Validate the required template keys
        missing_template_keys = [
            key for key in required_template_keys if key not in template
        ]
        if missing_template_keys:
            raise ValueError(f"Missing required template keys: {missing_template_keys}")

        # Validate the rest of the template keys
        for template_key in template:
            if template_key == "method":
                continue

            allowed_placeholders: Set[str] = set()
            used_placeholders = {
                name
                for _, name, _, _ in formatter.parse(self.template[template_key])
                if name is not None
            }
            if template_key in required_template_keys:
                allowed_placeholders.update(
                    required_template_keys[template_key]["required"]
                    + required_template_keys[template_key]["allowed"]
                )
                missing = (
                    set(required_template_keys[template_key]["required"])
                    - used_placeholders
                )
            elif template_key in allowed_template_keys:
                allowed_placeholders.update(
                    allowed_template_keys[template_key]["required"]
                    + allowed_template_keys[template_key]["allowed"]
                )
                missing = (
                    set(allowed_template_keys[template_key]["required"])
                    - used_placeholders
                )
            else:
                raise ValueError(f"Unsupported template section: {template_key}")

            if missing:
                raise ValueError(
                    f"Missing placeholders in {template_key} section: {missing}"
                )
            unsupported_placeholders = used_placeholders - allowed_placeholders
            if unsupported_placeholders:
                msg = f"Unsupported placeholders in {template_key} section: {unsupported_placeholders}"
                if strict:
                    raise ValueError(msg)

        print("Template validated successfully!")

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def _convert_doc_to_prompt_content(
        self, doc: Dict[str, Any], max_length: int
    ) -> str:
        if "text" in doc:
            content = doc["text"]
        elif "segment" in doc:
            content = doc["segment"]
        elif "contents" in doc:
            content = doc["contents"]
        elif "content" in doc:
            content = doc["content"]
        elif "body" in doc:
            content = doc["body"]
        else:
            content = doc["passage"]
        if "title" in doc and doc["title"]:
            content = "Title: " + doc["title"] + " " + "Content: " + content
        content = content.strip()
        content = fix_text(content)
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = " ".join(content.split()[: int(max_length)])

        return self._replace_number(content)

    def _generate_prefix_suffix(
        self, num: int, query: str, **kwargs: Any
    ) -> Tuple[str, str]:
        formatter = Formatter()
        prefix_placeholders = [
            name
            for _, name, _, _ in formatter.parse(self.template["prefix"])
            if name is not None
        ]
        suffix_placeholders = [
            name
            for _, name, _, _ in formatter.parse(self.template["suffix"])
            if name is not None
        ]

        prefix_fmt_values = suffix_fmt_values = {"num": num, "query": query}

        if "psg_ids" in suffix_placeholders:
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
        self, result: Result, rank_start: int, rank_end: int, use_alpha: bool
    ) -> str:
        max_length = 300 * (20 // (rank_end - rank_start))
        rank = 0
        body_text = ""

        for cand in result.candidates[rank_start:rank_end]:
            rank += 1
            content = self._convert_doc_to_prompt_content(cand.doc, max_length)

            identifier = chr(ALPH_START_IDX + rank) if use_alpha else str(rank)

            content = self._replace_number(content)
            fmt_values = {"rank": identifier, "candidate": content}
            single_text = self._format_template(
                template_key="body", fmt_values=fmt_values
            )

            body_text += f"{single_text}\n"

        return body_text

    def generate_prompt(self, result: Result, **kwargs: Any) -> List[Dict[str, str]]:
        try:
            rank_start = kwargs["rank_start"]
            rank_end = kwargs["rank_end"]
            use_alpha = kwargs.get("use_alpha", False)
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        query = result.query.text
        query = self._replace_number(query)
        num = len(result.candidates[rank_start:rank_end])

        prompt_messages = list()
        system_message = self.template.get("system_message")
        if system_message:
            prompt_messages.append({"role": "system", "content": system_message})

        prefix_text, suffix_text = self._generate_prefix_suffix(num, query)
        body_text = self._generate_body(result, rank_start, rank_end, use_alpha)
        prompt_text = ""

        if prefix_text:
            prompt_text += f"{prefix_text}\n"
        prompt_text += body_text
        if suffix_text:
            prompt_text += suffix_text

        prompt_messages.append({"role": "user", "content": prompt_text})

        return prompt_messages

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

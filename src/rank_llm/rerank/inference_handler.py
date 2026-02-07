import re
from abc import ABC, abstractmethod
from string import Formatter
from typing import Any, Dict, List

from ftfy import fix_text

from rank_llm.data import Candidate, Result, TemplateSectionConfig


class BaseInferenceHandler(ABC):
    def __init__(self, template: Dict[str, str]):
        self._formatter = Formatter()
        self._validate_template(template=template)
        self.template = template
        print("Template validated successfully!")

    @abstractmethod
    def _validate_template(self, template: Dict[str, str]):
        """
        Validates the structure and content of the provided template dictionary.

        Args:
            template (Dict[str, str]): Dictionary containing the prompt template parts (prefix, suffix, body, etc.)

        Raises:
            ValueError: If the template is missing required sections or contains invalid keywords
            TypeError: If any template part has incorrect type

        Note:
            Should check for:
            - Required sections (e.g., 'prefix', 'suffix', 'body')
            - Allowed keywords in each section
            - Proper string formatting
        """
        pass

    def _general_validation(
        self,
        template: Dict[str, str],
        template_section: Dict[str, TemplateSectionConfig],
        check_query: bool = False,
        strict: bool = False,
    ):
        # Validate the required template keys
        missing_template_keys = [
            key
            for key, config in template_section.items()
            if key not in template and config.required
        ]
        if missing_template_keys:
            raise ValueError(f"Missing required template keys: {missing_template_keys}")

        if check_query:
            query_present = (
                False if "prefix" in template or "suffix" in template else True
            )

        # Validate the rest of the template keys
        for template_key, template_text in template.items():
            if template_key not in template_section:
                raise ValueError(f"Unsupported template section: {template_key}")

            section = template_section[template_key]
            required_placeholders = section.required_placeholders
            allowed_placeholders = required_placeholders | section.allowed_placeholders
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

            if check_query and "query" in used_placeholders:
                query_present = True

        if check_query and not query_present:
            raise ValueError(
                "query placeholder must be present in prefix and/or suffix"
            )

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def _convert_doc_to_prompt_content(
        self, candidate: Candidate, max_length: int
    ) -> str:
        content = candidate.get_content()
        if "title" in candidate.doc and candidate.doc["title"]:
            content = "Title: " + candidate.doc["title"] + " " + "Content: " + content
        content = content.strip()
        content = fix_text(content)
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = " ".join(content.split()[: int(max_length)])

        return self._replace_number(content)

    def _format_template(
        self, template_key: str, fmt_values: Dict[str, str | int], verbose: bool = False
    ) -> str:
        """
        Replaces placeholder keywords in a template section with actual content.

        Args:
            template_key (str): The template section containing placeholders
            fmt_values (Dict[str, str | int]): Key-value pairs where keywords match the content for them
                (e.g., {"query": "actual query text", "num": 5})

        Returns:
            The template text with all keywords replaced by their values

        Example:
            If template_key = "prefix"
            and replacements = {"query": "llm ranking", "num": 3}
            Returns: "Query: llm ranking\nNumber: 3"
        """
        template_text = self.template.get(template_key, "")

        if not template_text:
            if verbose:
                print(
                    f"{template_key} is not a part of the template provided, setting {template_key} part as empty string"
                )
            return ""

        return template_text.format(**fmt_values)

    @abstractmethod
    def _generate_fewshot_prompt(
        self,
        num_examples: int = 0,
        examples: List[Dict[str, List[Dict[str, str]]]] = [],
    ) -> List[Dict[str, str]] | str:
        """
        Generates a few-shot prompt based on the provided examples.
        Args:
            num_examples (int): Number of examples to include in the few-shot prompt
            examples (List[Dict[str, List[Dict[str, str]]]]): List of example conversations
        Returns:
            A list of fewshot prompt messages or a single prompt message string.
        """
        pass

    @abstractmethod
    def generate_prompt(
        self, result: Result, **kwargs: Any
    ) -> List[Dict[str, str]] | str:
        """
        Generates complete prompt(s) for the ranking task.

        Args:
            result (Result): The search Result object to generate prompts for

        Returns:
            Tuple[str, int]: A tuple object containing the text response and the number of tokens in the response.

        Note:
            Should:
            1. Call _generate_prefix(), _generate_body(), and _generate_suffix()
            2. Combine all parts into final prompt(s)
            3. Calculate token counts for each prompt
            4. Handle both single and batched prompt cases
        """
        pass

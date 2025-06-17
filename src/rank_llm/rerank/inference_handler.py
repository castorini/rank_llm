from abc import ABC, abstractmethod
from typing import Any, Dict, List

from rank_llm.data import Result


class BaseInferenceHandler(ABC):
    def __init__(self, template: Dict[str, str]):
        self._validate_template(template=template)
        self.template = template

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

    def _format_template(
        self, template_key: str, fmt_values: Dict[str, str | int]
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
            print(
                f"{template_key} is not a part of the template provided, setting {template_key} part as empty string"
            )
            return ""

        return template_text.format(**fmt_values)

    @abstractmethod
    def generate_prompt(self, result: Result, **kwargs: Any) -> List[Dict[str, str]]:
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

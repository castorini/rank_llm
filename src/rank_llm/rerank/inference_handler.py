from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from rank_llm.data import Result


class BaseInferenceHandler(ABC):
    def __init__(self, template: Dict[str, str]):
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

    def _replace_key(self, template_part: str, **kwargs: Any) -> str:
        """
        Replaces placeholder keywords in a template section with actual content.

        Args:
            template_part (str): The template section containing placeholders
            **kwargs (Any): Key-value pairs where keys match template placeholders
                (e.g., {query: "actual query text", num: 5})

        Returns:
            The template string with all placeholders replaced by their values

        Example:
            If template_part = "Query: {query}\nNumber: {num}"
            and kwargs = {"query": "llm ranking", "num": 3}
            Returns: "Query: llm ranking\nNumber: 3"
        """
        pass

    @abstractmethod
    def generate_prompt(
        self, result: Result, **kwargs: Any
    ) -> Tuple[str, int] | List[Tuple[Dict[str, str], int]]:
        """
        Generates complete prompt(s) for the ranking task.

        Args:
            result (Result): The search Result object to generate prompts for
            rank_start (int): Starting index for documents to include
            rank_end (int): Ending index for documents to include

        Returns:
            Either:
            - A single tuple of (prompt_text, token_count)
            OR
            - List of tuples where each tuple contains (prompt_dict, token_count)
              for batched prompts

        Note:
            Should:
            1. Call _generate_prefix(), _generate_body(), and _generate_suffix()
            2. Combine all parts into final prompt(s)
            3. Calculate token counts for each prompt
            4. Handle both single and batched prompt cases
        """
        pass

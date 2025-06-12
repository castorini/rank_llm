from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

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

    @abstractmethod
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
    def _generate_prefix(
        self, num: Optional[int] = None, query: Optional[str] = None
    ) -> str:
        """
        Generates the prefix portion of the prompt using the template.

        Args:
            num (Optional[int] = None): Optional number to include in the prefix (e.g., number of results)
            query (Optional[str] = None): Optional query text to include in the prefix

        Returns:
            The fully formatted prefix string with all placeholders replaced

        Note:
            Should call _replace_key() with the prefix template and provided arguments
        """
        pass

    @abstractmethod
    def _generate_suffix(
        self, num: Optional[int] = None, query: Optional[int] = None
    ) -> str:
        """
        Generates the suffix portion of the prompt using the template.

        Args:
            num (Optional[int] = None): Optional number to include in the suffix (e.g., expected output count)
            query (Optional[str] = None): Optional query reference to include in the suffix

        Returns:
            The fully formatted suffix string with all placeholders replaced

        Note:
            Should call _replace_key() with the suffix template and provided arguments
        """
        pass

    @abstractmethod
    def _generate_body(self, result: Result) -> str:
        """
        Generates the main body content from a search result.

        Args:
            result (Result): The search Result object

        Returns:
            The fully formatted body string with all result documents processed

        Note:
            Should:
            1. Process each document in the result iteratively
            2. Call _replace_key() for each document using the body template
            3. Combine all processed documents into the final body string
        """
        pass

    @abstractmethod
    def generate_prompt(
        self, result: Result, rank_start: int, rank_end: int, batch_size: int
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

    @abstractmethod
    def response_analyzer(self, response: str):
        """More details on this later on"""
        pass

    def _clean_response(self, response: str, use_alpha: bool = False) -> str:
        ALPH_START_IDX = ord("A") - 1

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

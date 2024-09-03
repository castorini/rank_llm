from enum import Enum
from typing import Dict, List, Union


class Prompt(Enum):
    """
    Enum for choosing LLM prompts
    """

    UNSPECIFIED = "unspecified"
    RANK_GPT = "rank_GPT"
    RANK_GPT_APEER = "rank_GPT_APEER"
    LRL = "LRL"
    MONOT5 = "monot5"
    LiT5 = "LiT5"

    def __str__(self):
        return self.value

    def prefix(self, *args, **kwargs) -> Union[str, List[Dict[str, str]]]:
        """
        returns the prompt prefix
        """

        try:
            query = args[0]
            top_k = args[1]
        except Exception as e:
            raise ValueError("Missing query or top_k positional arguments")

        match self:
            case self.RANK_GPT:
                # rank listwise OS LLM uses a slight variation of the Rank GPT prompt
                is_os_llm = kwargs.get("is_os_llm", False)
                if is_os_llm:
                    return f"I will provide you with {top_k} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
                else:
                    return [
                        {
                            "role": "system",
                            "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
                        },
                        {
                            "role": "user",
                            "content": f"I will provide you with {top_k} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
                        },
                        {
                            "role": "assistant",
                            "content": "Okay, please provide the passages.",
                        },
                    ]
            case self.RANK_GPT_APEER:
                return [
                    {
                        "role": "system",
                        "content": "As RankGPT, your task is to evaluate and rank unique passages based on their relevance and accuracy to a given query. Prioritize passages that directly address the query and provide detailed, correct answers. Ignore factors such as length, complexity, or writing style unless they seriously hinder readability.",
                    },
                    {
                        "role": "user",
                        "content": f"In response to the query: [querystart] {query} [queryend], rank the passages. Ignore aspects like length, complexity, or writing style, and concentrate on passages that provide a comprehensive understanding of the query. Take into account any inaccuracies or vagueness in the passages when determining their relevance.",
                    },
                ]
            case _:
                raise ValueError(
                    f"Prefix for prompt mode {self.__str__()} is unsupported."
                )

    def suffix(self, **kwargs) -> str:
        """
        returns the prompt suffix
        """

        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", 0)

        match self:
            case self.RANK_GPT:
                is_os_llm = kwargs.get("is_os_llm", False)
                if is_os_llm:
                    variable_passages = kwargs.get("variable_passages", False)
                    example_ordering = "[2] > [1]" if variable_passages else "[4] > [2]"
                    return f"Search Query: {query}.\nRank the {top_k} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."
                else:
                    return f"Search Query: {query}. \nRank the {top_k} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."
            case self.RANK_GPT_APEER:
                return f"Given the query: [querystart] {query} [queryend], produce a succinct and clear ranking of all passages, from most to least relevant, using their identifiers. The format should be [rankstart] [most relevant passage ID] > [next most relevant passage ID] > ... > [least relevant passage ID] [rankend]. Refrain from including any additional commentary or explanations in your ranking."
            case _:
                raise ValueError(
                    f"Suffix for prompt mode {self.__str__()} is unsupported."
                )

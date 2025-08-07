import re
from typing import Any, Dict, List, Union

try:
    from transformers import T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    T5Tokenizer = None

from rank_llm.data import Result, TemplateSectionConfig
from rank_llm.rerank.inference_handler import BaseInferenceHandler


class PairwiseInferenceHandler(BaseInferenceHandler):
    def __init__(self, template: Dict[str, str]):
        super().__init__(template)

    def _validate_template(self, template: Dict[str, str], strict: bool = False):
        TEMPLATE_SECTIONS: Dict[str, TemplateSectionConfig] = {
            "method": TemplateSectionConfig(
                required=True,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
            "body": TemplateSectionConfig(
                required=True,
                required_placeholders={"query", "doc1", "doc2"},
                allowed_placeholders=set(),
            ),
        }

        # Validate the method value
        if template["method"] != "pairwise":
            raise ValueError(
                f'Incorrect method type, expected "pairwise", got {template["method"]}'
            )

        self._general_validation(
            template=template, template_section=TEMPLATE_SECTIONS, strict=strict
        )

    # TODO (issue #273): May need to add prefix/suffix generation function later

    def _generate_fewshot_prompt(
        self,
        num_examples: int = 0,
        examples: List[Dict[str, List[Dict[str, str]]]] = [],
    ) -> str:
        text_examples = []
        pattern = re.compile(
            r"Query: (?P<query>.+?) Document0: (?P<doc0>.+?) Document1: (?P<doc1>.+)$"
        )

        for ex in examples[: min(num_examples, len(examples))]:
            try:
                # assume each value for conversation contain 2 values (user query + docs, asssistant response)
                user_msg = ex["conversations"][0]["value"]

                match = pattern.match(user_msg)
                if not match:
                    continue

                example_query = match.group("query").strip()
                example_doc0 = match.group("doc0").strip()
                example_doc1 = match.group("doc1").strip()
                example_relevance = ex["conversations"][1]["value"].strip()

                fmt_values = {
                    "query": example_query,
                    "doc1": example_doc0,
                    "doc2": example_doc1,
                }
                fewshot_text = self._format_template(
                    template_key="body", fmt_values=fmt_values
                )

                keyword = "Relevant"
                if "Relevance" in fewshot_text:
                    keyword = "Relevance"

                if keyword in fewshot_text:
                    pos = fewshot_text.rfind(keyword)
                    fewshot_text = (
                        fewshot_text[: pos + len(keyword)]
                        + f": {example_relevance}"
                        + fewshot_text[pos + len(keyword) :]
                    )
                else:
                    fewshot_text += f" {keyword}: {example_relevance}"

                text_examples.append(fewshot_text.strip())
            except (KeyError, IndexError):
                continue

        return "\n".join(text_examples) + "\n\n" if text_examples else ""

    def _generate_body(
        self,
        result: Result,
        index1: int,
        index2: int,
        single_doc_max_token: int,
        tokenizer: Union["T5Tokenizer", Any],
    ) -> str:
        doc1_raw = self._convert_doc_to_prompt_content(
            result.candidates[index1].doc, max_length=single_doc_max_token
        )
        doc2_raw = self._convert_doc_to_prompt_content(
            result.candidates[index2].doc, max_length=single_doc_max_token
        )

        doc1_tokens = tokenizer.encode(
            doc1_raw, truncation=True, max_length=single_doc_max_token
        )
        doc2_tokens = tokenizer.encode(
            doc2_raw, truncation=True, max_length=single_doc_max_token
        )

        query = self._replace_number(result.query.text)
        doc1 = tokenizer.decode(doc1_tokens, skip_special_tokens=True)
        doc2 = tokenizer.decode(doc2_tokens, skip_special_tokens=True)

        fmt_values = {"query": query, "doc1": doc1, "doc2": doc2}
        body_text = self._format_template(template_key="body", fmt_values=fmt_values)

        return body_text

    def generate_prompt(self, result: Result, **kwargs: Any) -> str:
        try:
            index1 = kwargs["index1"]
            index2 = kwargs["index2"]
            max_token = kwargs["max_token"]
            tokenizer = kwargs["tokenizer"]
            num_fewshot_examples = kwargs.get("num_fewshot_examples", 0)
            fewshot_examples = kwargs.get("fewshot_examples", [])
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        single_doc_max_token = max_token // 2

        prompt = ""
        if num_fewshot_examples > 0 and fewshot_examples:
            prompt = self._generate_fewshot_prompt(
                num_examples=num_fewshot_examples,
                examples=fewshot_examples,
            )
        prompt += self._generate_body(
            result=result,
            index1=index1,
            index2=index2,
            single_doc_max_token=single_doc_max_token,
            tokenizer=tokenizer,
        )
        return prompt.replace("<unk>", "")

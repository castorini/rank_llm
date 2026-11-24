import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import T5Tokenizer
else:
    T5Tokenizer = Any

from rank_llm.data import Result, TemplateSectionConfig
from rank_llm.rerank.inference_handler import BaseInferenceHandler, compose_query_block

# Optional template key naming the tokens a pointwise judge answers with:
#   output_tokens:
#     positive: ["yes"]
#     negative: ["no"]
# The prompt tells the model how to answer, so the answer tokens are declared
# beside it. Spellings are matched as token ids (quote leading spaces in YAML,
# e.g. " yes"); see PointwiseAnswerTokens.
OUTPUT_TOKENS_KEY = "output_tokens"


def validate_output_tokens(value: Any) -> None:
    """Raise unless ``value`` maps positive and negative to non-empty lists of strings."""
    if not isinstance(value, dict) or set(value) != {"positive", "negative"}:
        raise ValueError(
            f"'{OUTPUT_TOKENS_KEY}' must have exactly the keys 'positive' and "
            f"'negative', each a list of answer token spellings; got {value!r}"
        )
    for side in ("positive", "negative"):
        spellings = value[side]
        if (
            not isinstance(spellings, list)
            or not spellings
            or not all(isinstance(s, str) and s for s in spellings)
        ):
            raise ValueError(
                f"'{OUTPUT_TOKENS_KEY}.{side}' must be a non-empty list of "
                f"non-empty strings; got {spellings!r}"
            )


class PointwiseInferenceHandler(BaseInferenceHandler):
    def __init__(self, template: dict[str, str]):
        super().__init__(template)

    def _validate_template(self, template: dict[str, str], strict: bool = False):
        if OUTPUT_TOKENS_KEY in template:
            # Structured, not a format string: checked here, not as a section.
            validate_output_tokens(template[OUTPUT_TOKENS_KEY])
            template = {k: v for k, v in template.items() if k != OUTPUT_TOKENS_KEY}
        is_reranker = template.get("message_roles") == "reranker"
        if "query_block" in template and not is_reranker:
            raise ValueError(
                "'query_block' composes the query role message and needs "
                "message_roles: reranker; without it only 'body' is rendered"
            )

        TEMPLATE_SECTIONS = {
            "method": TemplateSectionConfig(
                required=True,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
            "system_message": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
            "body": TemplateSectionConfig(
                required=not is_reranker,
                required_placeholders=(
                    set() if is_reranker else {"query", "doc_content"}
                ),
                allowed_placeholders=(
                    {"instruction", "query", "doc_content"} if is_reranker else set()
                ),
                accepts_extra_fields=True,
            ),
            "message_roles": TemplateSectionConfig(
                required=False,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
            "instruction": TemplateSectionConfig(
                required=is_reranker,
                required_placeholders=set(),
                allowed_placeholders=set(),
            ),
            # Optional: composes the text sent as the `query` role, so a
            # template can fold declared extra_fields into the query block
            # instead of requiring the caller to pre-assemble the string.
            "query_block": TemplateSectionConfig(
                required=False,
                required_placeholders={"query"},
                allowed_placeholders=set(),
                accepts_extra_fields=True,
            ),
        }

        # Validate the method value
        if template["method"] != "pointwise":
            raise ValueError(
                f'Incorrect method type, expected "pointwise", got {template["method"]}'
            )

        self._general_validation(
            template=template, template_section=TEMPLATE_SECTIONS, strict=strict
        )

    # TODO (issue #273): May need to add prefix/suffix generation function later
    def _generate_fewshot_prompt(
        self,
        num_examples: int = 0,
        examples: list[dict[str, list[dict[str, str]]]] | None = None,
        **kwargs: Any,
    ) -> str:
        if examples is None:
            examples = []
        text_examples = []
        pattern = re.compile(r"Query: (?P<query>.+?) Document: (?P<doc>.+)$")

        for ex in examples[: min(num_examples, len(examples))]:
            try:
                # assume each value to conversation key have at least 2 values (user: query + doc, assistant: score of relevance)
                user_msg = ex["conversations"][0]["value"]

                match = pattern.match(user_msg)
                if not match:
                    continue

                example_query = match.group("query").strip()
                example_doc = match.group("doc").strip()
                example_relevance = ex["conversations"][1]["value"].strip()

                # Examples come from other queries and carry no extra fields.
                fmt_values = {
                    "query": example_query,
                    "doc_content": example_doc,
                    **{name: "" for name in self._declared_extra_fields(self.template)},
                }
                fewshot_text = self._format_template("body", fmt_values=fmt_values)

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
        index: int,
        max_doc_tokens: int,
        tokenizer: T5Tokenizer,
        max_passage_words: int | None = None,
    ) -> str:
        query = self._replace_number(result.query.text)
        word_limit = (
            max_passage_words if max_passage_words is not None else max_doc_tokens
        )
        doc_raw = self._convert_doc_to_prompt_content(
            result.candidates[index].doc, max_length=word_limit
        )
        doc_tokens = tokenizer.encode(
            doc_raw, truncation=True, max_length=max_doc_tokens
        )
        doc = tokenizer.decode(doc_tokens, skip_special_tokens=True)

        fmt_values = {
            "query": query,
            "doc_content": doc,
            **self._extra_field_values(result),
        }
        body_text = self._format_template(template_key="body", fmt_values=fmt_values)

        return body_text

    def generate_prompt(self, result: Result, **kwargs: Any) -> str:
        try:
            index = kwargs["index"]
            max_doc_tokens = kwargs["max_doc_tokens"]
            tokenizer = kwargs["tokenizer"]
            num_fewshot_examples = kwargs.get("num_fewshot_examples", 0)
            fewshot_examples = kwargs.get("fewshot_examples", [])
            max_passage_words = kwargs.get("max_passage_words", None)
        except KeyError as err:
            raise ValueError(f"Missing required parameter: {err}") from err

        prompt = ""
        if num_fewshot_examples > 0 and fewshot_examples:
            prompt = self._generate_fewshot_prompt(
                num_examples=num_fewshot_examples, examples=fewshot_examples
            )
        prompt += self._generate_body(
            result=result,
            index=index,
            max_doc_tokens=max_doc_tokens,
            tokenizer=tokenizer,
            max_passage_words=max_passage_words,
        )
        return prompt.replace("<unk>", "")

    def _build_query_role_content(self, result: Result, query: str) -> str:
        """Text for the `query` role message.

        Without a `query_block` section this is the query text, matching the
        stock Qwen3-Reranker layout. With one, the section composes the query
        together with the template's declared extra_fields. Template lines whose
        only substitutions resolve to empty are dropped first, so one template
        degrades cleanly when optional context is absent.
        """
        if "query_block" not in self.template:
            return query
        return compose_query_block(
            self.template["query_block"], query, self._extra_field_values(result)
        )

    def generate_chat_messages(
        self, result: Result, **kwargs: Any
    ) -> list[dict[str, str]]:
        """Build chat messages for pointwise VLLM backends.

        When ``message_roles`` is ``"reranker"`` in the template, emits separate
        ``query`` / ``document`` role messages expected by models like
        Qwen3-Reranker whose chat template maps those roles to named fields.
        Otherwise falls back to the standard ``system`` + ``user`` layout.
        """
        try:
            index = kwargs["index"]
            max_doc_tokens = kwargs["max_doc_tokens"]
            tokenizer = kwargs["tokenizer"]
            num_fewshot_examples = kwargs.get("num_fewshot_examples", 0)
            fewshot_examples = kwargs.get("fewshot_examples", [])
            max_passage_words = kwargs.get("max_passage_words", None)
        except KeyError as err:
            raise ValueError(f"Missing required parameter: {err}") from err

        query = self._replace_number(result.query.text)
        word_limit = (
            max_passage_words if max_passage_words is not None else max_doc_tokens
        )
        doc_raw = self._convert_doc_to_prompt_content(
            result.candidates[index].doc, max_length=word_limit
        )
        doc_tokens = tokenizer.encode(
            doc_raw, truncation=True, max_length=max_doc_tokens
        )
        doc = tokenizer.decode(doc_tokens, skip_special_tokens=True)

        if self.template.get("message_roles") == "reranker":
            instruction = self.template.get("instruction", "").strip()
            return [
                {"role": "system", "content": instruction},
                {
                    "role": "query",
                    "content": self._build_query_role_content(result, query),
                },
                {"role": "document", "content": doc},
            ]

        user_parts: list[str] = []
        if num_fewshot_examples > 0 and fewshot_examples:
            user_parts.append(
                self._generate_fewshot_prompt(
                    num_examples=num_fewshot_examples, examples=fewshot_examples
                )
            )
        fmt_values = {
            "query": query,
            "doc_content": doc,
            **self._extra_field_values(result),
        }
        body_text = self._format_template(template_key="body", fmt_values=fmt_values)
        user_parts.append(body_text)
        user_content = "".join(user_parts).replace("<unk>", "")

        messages: list[dict[str, str]] = []
        system_msg = self.template.get("system_message")
        if system_msg:
            messages.append({"role": "system", "content": system_msg.strip()})
        messages.append({"role": "user", "content": user_content})
        return messages

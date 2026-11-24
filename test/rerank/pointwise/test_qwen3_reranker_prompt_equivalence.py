"""The reranker-roles chat template must reproduce Qwen3-Reranker's own prompt.

`message_roles: "reranker"` emits system/query/document messages and lets the
model's chat template assemble them. That is only correct if the result is
byte-identical to the prefix/middle/suffix layout published on the Qwen3-Reranker
model card -- otherwise scores would be computed against a prompt the model was
never trained on. These tests pin that down for the stock template and for the
ACARe context modes built on top of it.

Needs the tokenizers from the Hub, so it is skipped when they are unavailable.
"""

import unittest
from importlib.resources import files

import yaml

TEMPLATES = files("rank_llm.rerank.prompt_templates")

QWEN = "Qwen/Qwen3-Reranker-4B"

SYSTEM_MESSAGE = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
PREFIX = f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

SUB_QUERY = "St John's Health Center celebrity death"
GLOBAL_QUERY = "What's the name of the hospital they died?"
REASONING = "Need the hospital where the Grammy-winning entertainer died."
DOCUMENT = "The singer died at St. John's Health Center in Santa Monica."


def load_tokenizer(model):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, padding_side="left")
    except Exception as exc:  # network, auth, or transformers missing
        raise unittest.SkipTest(f"tokenizer for {model} unavailable: {exc}") from exc


def model_card_prompt(instruction, query_block, document):
    """The published Qwen3-Reranker recipe: prefix + formatted middle + suffix."""
    middle = (
        f"<Instruct>: {instruction}\n<Query>: {query_block}\n<Document>: {document}"
    )
    return PREFIX + middle + SUFFIX


def chat_template_prompt(tokenizer, instruction, query_block, document):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "query", "content": query_block},
        {"role": "document", "content": document},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    ids = tokenizer.apply_chat_template(messages, tokenize=True)["input_ids"]
    return text, ids


def load(name):
    with (TEMPLATES / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TestQwen3RerankerPromptEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer(QWEN)

    def assert_matches_model_card(self, instruction, query_block):
        expected = model_card_prompt(instruction, query_block, DOCUMENT)
        text, ids = chat_template_prompt(
            self.tokenizer, instruction, query_block, DOCUMENT
        )
        self.assertEqual(text, expected)
        expected_ids = (
            self.tokenizer.encode(PREFIX, add_special_tokens=False)
            + self.tokenizer.encode(
                f"<Instruct>: {instruction}\n<Query>: {query_block}\n"
                f"<Document>: {DOCUMENT}",
                add_special_tokens=False,
            )
            + self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        )
        self.assertEqual(ids, expected_ids)

    def test_stock_template_reproduces_the_model_card_prompt(self):
        self.assert_matches_model_card(
            load("pointwise_vllm_template.yaml")["instruction"].strip(), SUB_QUERY
        )

    def test_acare_m1(self):
        self.assert_matches_model_card(
            load("acare_m1_template.yaml")["instruction"],
            f"#Reasoning Trace: {REASONING}\n#Query: {SUB_QUERY}",
        )

    def test_acare_m2(self):
        self.assert_matches_model_card(
            load("acare_m2_template.yaml")["instruction"],
            f"#Original Question (global query): {GLOBAL_QUERY}\n#Query: {SUB_QUERY}",
        )

    def test_acare_m3(self):
        self.assert_matches_model_card(
            load("acare_m3_template.yaml")["instruction"],
            f"#Original Question (global query): {GLOBAL_QUERY}\n"
            f"#Reasoning Trace: {REASONING}\n#Query: {SUB_QUERY}",
        )


if __name__ == "__main__":
    unittest.main()

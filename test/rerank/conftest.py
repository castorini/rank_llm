class FakeTokenizer:
    """Lightweight stand-in for T5Tokenizer used in unit tests."""

    def encode(self, text, truncation=True, max_length=None):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)

import logging
from importlib import import_module

from .api_keys import (
    get_azure_openai_args,
    get_genai_api_key,
    get_openai_api_key,
    get_openrouter_api_key,
)
from .identity_reranker import IdentityReranker
from .rankllm import RankLLM

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

__all__ = [
    "IdentityReranker",
    "RankLLM",
    "get_azure_openai_args",
    "get_openai_api_key",
    "get_genai_api_key",
    "get_openrouter_api_key",
    "Reranker",
]


def __getattr__(name):
    if name != "Reranker":
        raise AttributeError(f"module {__name__} has no attribute {name}")
    value = getattr(import_module(".reranker", __name__), name)
    globals()[name] = value
    return value

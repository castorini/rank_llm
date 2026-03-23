from importlib import import_module

from .retrieval_method import RetrievalMethod
from .retriever import RetrievalMode, Retriever
from .service_retriever import ServiceRetriever
from .topics_dict import TOPICS
from .utils import download_cached_hits

__all__ = [
    "INDICES",
    "TOPICS",
    "RetrievalMethod",
    "PyseriniRetriever",
    "ServiceRetriever",
    "Retriever",
    "RetrievalMode",
    "download_cached_hits",
]


def __getattr__(name: str):
    if name == "INDICES":
        module = import_module("rank_llm.retrieve.indices_dict")
    elif name == "PyseriniRetriever":
        module = import_module("rank_llm.retrieve.pyserini_retriever")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value

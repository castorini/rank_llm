from importlib import import_module
from typing import Any

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

_MODULE_BY_SYMBOL = {
    "INDICES": "rank_llm.retrieve.indices_dict",
    "TOPICS": "rank_llm.retrieve.topics_dict",
    "RetrievalMethod": "rank_llm.retrieve.retrieval_method",
    "RetrievalMode": "rank_llm.retrieve.retriever",
    "Retriever": "rank_llm.retrieve.retriever",
    "PyseriniRetriever": "rank_llm.retrieve.pyserini_retriever",
    "ServiceRetriever": "rank_llm.retrieve.service_retriever",
    "download_cached_hits": "rank_llm.retrieve.utils",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULE_BY_SYMBOL:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_MODULE_BY_SYMBOL[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

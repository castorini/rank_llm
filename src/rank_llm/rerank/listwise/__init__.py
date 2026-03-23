from importlib import import_module

__all__ = [
    "RankListwiseOSLLM",
    "VicunaReranker",
    "ZephyrReranker",
    "SafeOpenai",
    "SafeGenai",
]

_MODULE_BY_SYMBOL = {
    "RankListwiseOSLLM": "rank_llm.rerank.listwise.rank_listwise_os_llm",
    "SafeGenai": "rank_llm.rerank.listwise.rank_gemini",
    "SafeOpenai": "rank_llm.rerank.listwise.rank_gpt",
    "VicunaReranker": "rank_llm.rerank.listwise.vicuna_reranker",
    "ZephyrReranker": "rank_llm.rerank.listwise.zephyr_reranker",
}


def __getattr__(name: str):
    if name not in _MODULE_BY_SYMBOL:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_MODULE_BY_SYMBOL[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

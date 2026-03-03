from importlib import import_module

__all__ = [
    "RankListwiseOSLLM",
    "VicunaReranker",
    "ZephyrReranker",
    "SafeOpenai",
    "SafeGenai",
]


_LAZY_IMPORTS = {
    "RankListwiseOSLLM": ".rank_listwise_os_llm",
    "VicunaReranker": ".vicuna_reranker",
    "ZephyrReranker": ".zephyr_reranker",
    "SafeOpenai": ".rank_gpt",
    "SafeGenai": ".rank_gemini",
}


def __getattr__(name):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    try:
        module = import_module(module_path, __name__)
    except ImportError as e:
        if name in {"RankListwiseOSLLM", "VicunaReranker", "ZephyrReranker"}:
            raise ImportError(
                f"{name} requires optional dependencies. Install with "
                "`pip install \"rank-llm[vllm]\"`."
            ) from e
        if name == "SafeOpenai":
            raise ImportError(
                "SafeOpenai requires optional dependencies. Install with "
                "`pip install \"rank-llm[openai]\"`."
            ) from e
        if name == "SafeGenai":
            raise ImportError(
                "SafeGenai requires optional dependencies. Install with "
                "`pip install \"rank-llm[genai]\"`."
            ) from e
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value

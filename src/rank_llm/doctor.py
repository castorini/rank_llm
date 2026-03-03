import importlib.util
import os
import platform
import shutil
import sys


def _installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _print_capability(name: str, installed: bool, install_cmd: str) -> None:
    status = "OK" if installed else "MISSING"
    print(f"- {name}: {status}")
    if not installed:
        print(f"  fix: {install_cmd}")


def main() -> None:
    print("RankLLM doctor")
    print(f"- python: {platform.python_version()}")
    print(f"- executable: {sys.executable}")
    print(f"- uv: {'OK' if shutil.which('uv') else 'MISSING'}")
    if not shutil.which("uv"):
        print("  fix: curl -LsSf https://astral.sh/uv/install.sh | sh")

    print("")
    print("Optional backend checks:")
    _print_capability(
        "OpenAI backend", _installed("openai"), 'pip install "rank-llm[openai]"'
    )
    _print_capability(
        "vLLM backend", _installed("vllm"), 'pip install "rank-llm[vllm]"'
    )
    _print_capability(
        "Gemini backend",
        _installed("google.generativeai"),
        'pip install "rank-llm[genai]"',
    )
    _print_capability(
        "Pyserini retriever",
        _installed("pyserini"),
        'pip install "rank-llm[pyserini]"',
    )

    print("")
    print("Environment variables:")
    env_vars = [
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEN_AI_API_KEY",
        "AZURE_OPENAI_API_BASE",
        "AZURE_OPENAI_API_VERSION",
        "HF_HOME",
    ]
    for key in env_vars:
        print(f"- {key}: {'set' if os.getenv(key) else 'unset'}")


if __name__ == "__main__":
    main()

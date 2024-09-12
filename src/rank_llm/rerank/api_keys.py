import os
from typing import Dict

from dotenv import load_dotenv

# Common OpenAI API key paths
paths = [
    "OPENAI_API_KEY",
    "OPEN_AI_API_KEY",
]


def get_openai_api_key() -> str:
    load_dotenv(dotenv_path=f".env.local")

    for path in paths:
        if os.getenv(path) is not None:
            return os.getenv(path)
    return None


def get_azure_openai_args() -> Dict[str, str]:
    load_dotenv(dotenv_path=f".env.local")
    azure_args = {
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
    }

    # Sanity check
    assert all(
        list(azure_args.values())
    ), "Ensure that `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION` are set"
    return azure_args

"""
Helpers for merging user sampling overrides across OpenAI v1 SDK and vLLM.

OpenAI-compatible clients only accept a fixed set of sampling fields as
top-level kwargs; keys like repetition_penalty and top_k must be sent inside
extra_body when talking to vLLM.
"""

from __future__ import annotations

from typing import Any

# Parameters accepted as top-level arguments by openai.ChatCompletion APIs
# used with vLLM. Everything else from the user's JSON lands in ``extra_body``.
OPENAI_CHAT_COMPLETION_SAMPLING_KEYS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "seed",
        "logit_bias",
        "n",
    }
)

# Keys RankLLM controls for listwise decoding; ignore if present in user JSON.
RANKLISTWISE_OWNED_SAMPLING_KEYS: frozenset[str] = frozenset(
    {
        "max_tokens",
        "min_tokens",
        "max_completion_tokens",
        "max_new_tokens",
        "min_new_tokens",
        "logprobs",
        "top_logprobs",
        "chat_template_kwargs",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_grammar",
    }
)


def sanitize_sampling_kwargs(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Return a shallow copy with RankLLM-reserved keys removed."""
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k in RANKLISTWISE_OWNED_SAMPLING_KEYS:
            continue
        out[k] = v
    return out


def split_openai_chat_sampling(
    extras: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Partition ``extras`` into (direct_openai_kwargs, extra_body_fields).

    ``extra_body_fields`` merge into vLLM's HTTP extra_body beside
    ``chat_template_kwargs`` and other RankLLM defaults.
    """
    direct: dict[str, Any] = {}
    body: dict[str, Any] = {}
    for k, v in extras.items():
        if k in OPENAI_CHAT_COMPLETION_SAMPLING_KEYS:
            direct[k] = v
        else:
            body[k] = v
    return direct, body

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rank_llm.cli.spec import EXIT_CODES

_PROVIDER_MODULE_TOKENS = (
    "openai",
    "google",
    "cohere",
    "anthropic",
    "httpx",
    "requests",
)
_PROVIDER_MESSAGE_TOKENS = (
    "rate limit",
    "quota",
    "timeout",
    "timed out",
    "connection error",
    "api connection",
    "service unavailable",
    "too many requests",
    "bad gateway",
    "gateway timeout",
    "openrouter",
)
_PREREQUISITE_MESSAGE_TOKENS = (
    "please provide openai keys",
    "openai_api_key",
    "openrouter_api_key",
    "gen_ai_api_key",
    "azure_openai_api_base",
    "azure_openai_api_version",
    "install the `",
    "requires the ",
    "missing dependencies",
)


@dataclass(frozen=True)
class ErrorDescriptor:
    message: str
    status: str
    exit_code: int
    error_code: str
    details: dict[str, Any]
    retryable: bool = False


def classify_exception(error: Exception) -> ErrorDescriptor:
    message = str(error)
    normalized = message.lower()
    module_name = type(error).__module__.lower()

    if isinstance(error, FileNotFoundError):
        return ErrorDescriptor(
            message=message,
            status="validation_error",
            exit_code=EXIT_CODES["missing_resource"],
            error_code="missing_resource",
            details={},
        )

    if isinstance(error, (ImportError, ModuleNotFoundError, AssertionError)) or any(
        token in normalized for token in _PREREQUISITE_MESSAGE_TOKENS
    ):
        return ErrorDescriptor(
            message=message,
            status="validation_error",
            exit_code=EXIT_CODES["missing_prerequisite"],
            error_code="missing_prerequisite",
            details={},
        )

    if any(token in module_name for token in _PROVIDER_MODULE_TOKENS) or any(
        token in normalized for token in _PROVIDER_MESSAGE_TOKENS
    ):
        retryable = any(
            token in normalized
            for token in (
                "rate limit",
                "timeout",
                "timed out",
                "connection error",
                "api connection",
                "service unavailable",
                "too many requests",
                "bad gateway",
                "gateway timeout",
            )
        )
        return ErrorDescriptor(
            message=message,
            status="provider_error",
            exit_code=EXIT_CODES["provider_error"],
            error_code="provider_error",
            details={},
            retryable=retryable,
        )

    return ErrorDescriptor(
        message=message,
        status="runtime_error",
        exit_code=EXIT_CODES["runtime_error"],
        error_code="runtime_error",
        details={},
    )


def has_partial_success_metrics(metrics: dict[str, Any] | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    ok = metrics.get("ok")
    if not isinstance(ok, int) or ok <= 0:
        return False
    error_total = 0
    for key, value in metrics.items():
        if key == "ok" or not isinstance(value, int):
            continue
        error_total += value
    return error_total > 0

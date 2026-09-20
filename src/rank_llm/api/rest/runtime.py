from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rank_llm.api.adapters import make_data_artifact, serialize_data
from rank_llm.api.error_utils import classify_exception
from rank_llm.api.operations import run_rerank, run_retrieve_and_rerank
from rank_llm.api.options import (
    RerankOptions,
    RetrievalOptions,
    normalize_rerank_input,
    option_values,
    prepare_rerank_request,
)
from rank_llm.api.responses import CommandResponse
from rank_llm.api.spec import EXIT_CODES
from rank_llm.rerank import Reranker


@dataclass
class ServerConfig(RerankOptions):
    host: str = "0.0.0.0"
    port: int = 8082
    _reranker_cache: dict[tuple[tuple[str, Any], ...], Reranker] = field(
        default_factory=dict, init=False, repr=False
    )


def initialize_reranker(
    config: ServerConfig, effective_config: RerankOptions | None = None
) -> Reranker:
    effective_config = effective_config or config
    options = option_values(effective_config)
    # Output truncation does not change the initialized model.
    options.pop("top_k_rerank")
    cache_key = tuple(options.items())
    if cache_key not in config._reranker_cache:
        model_path = options.pop("model_path")
        for name in ("prompt_template_path", "few_shot_file", "base_url"):
            options[name] = options[name] or None
        config._reranker_cache[cache_key] = Reranker(
            Reranker.create_model_coordinator(model_path, None, False, **options)
        )
    return config._reranker_cache[cache_key]


def run_rerank_request(
    payload: dict[str, Any], *, config: ServerConfig, retrieval: bool = False
) -> CommandResponse:
    options, validation = prepare_rerank_request(
        payload, defaults=config, retrieval=retrieval
    )
    input_mode = (
        (
            "service"
            if payload.get("retriever_host")
            else "requests-file"
            if payload.get("requests_file")
            else "dataset"
        )
        if retrieval
        else "direct"
    )
    response = CommandResponse(
        command="rerank",
        validation=validation,
        inputs={"mode": input_mode, "transport": "http"},
        resolved={
            "model_path": options.model_path,
            "input_mode": input_mode,
            "transport": "http",
        },
    )
    reranker = initialize_reranker(config, options)
    if retrieval:
        results = run_retrieve_and_rerank(
            options=options,
            retrieval=RetrievalOptions(**option_values(payload, RetrievalOptions)),
            reranker=reranker,
        )
    else:
        results = run_rerank(
            options=options, **normalize_rerank_input(payload), reranker=reranker
        )
    response.artifacts = [make_data_artifact("rerank-results", serialize_data(results))]
    return response


def validation_error_response(message: str) -> CommandResponse:
    return CommandResponse(
        command="rerank",
        status="validation_error",
        exit_code=EXIT_CODES["validation_error"],
        errors=[
            {
                "code": "validation_error",
                "message": message,
                "details": {},
                "retryable": False,
            }
        ],
    )


def runtime_error_response(error: Exception) -> CommandResponse:
    descriptor = classify_exception(error)
    return CommandResponse(
        command="rerank",
        status=descriptor.status,
        exit_code=descriptor.exit_code,
        errors=[
            {
                "code": descriptor.error_code,
                "message": descriptor.message,
                "details": descriptor.details,
                "retryable": descriptor.retryable,
            }
        ],
    )

from __future__ import annotations

from dataclasses import dataclass, field, fields
from threading import Lock
from typing import Any

from rank_llm.api.adapters import make_data_artifact, serialize_data
from rank_llm.api.error_utils import classify_exception
from rank_llm.api.operations import run_rerank, run_retrieve_and_rerank
from rank_llm.api.options import (
    RerankOptions,
    RetrievalOptions,
    option_values,
    validate_option_values,
    validate_rerank_options,
    validate_retrieval_options,
)
from rank_llm.api.responses import CommandResponse
from rank_llm.api.spec import EXIT_CODES
from rank_llm.data import RerankValidationError, normalize_rerank_input
from rank_llm.rerank import Reranker
from rank_llm.utils import default_device


@dataclass
class ServerConfig(RerankOptions):
    host: str = "0.0.0.0"
    port: int = 8082
    _reranker_cache: dict[tuple[tuple[str, Any], ...], Reranker] = field(
        default_factory=dict, init=False, repr=False
    )

    _cache_lock: Any = field(default_factory=Lock, init=False, repr=False)


def initialize_reranker(
    config: ServerConfig, effective_config: RerankOptions | None = None
) -> Reranker:
    effective_config = effective_config or config
    options = effective_config.to_kwargs()
    # These settings affect execution/output, not model construction.
    for name in (
        "top_k_rerank",
        "num_passes",
        "shuffle_candidates",
        "print_prompts_responses",
        "populate_invocations_history",
    ):
        options.pop(name)
    cache_key = tuple(options.items())
    with config._cache_lock:
        if cache_key not in config._reranker_cache:
            model_path = options.pop("model_path")
            config._reranker_cache[cache_key] = Reranker(
                Reranker.create_model_coordinator(
                    model_path, None, False, device=default_device(), **options
                )
            )
        return config._reranker_cache[cache_key]


def run_rerank_request(
    payload: dict[str, Any], *, config: ServerConfig, retrieval: bool = False
) -> CommandResponse:
    if not isinstance(payload, dict):
        raise RerankValidationError("payload must be an object")
    allowed = (
        {f.name for f in fields(RetrievalOptions)}
        if retrieval
        else {"query", "candidates"}
    )
    unknown = sorted(set(payload) - allowed - {"overrides"})
    if unknown:
        raise RerankValidationError(
            "unsupported request field(s): " + ", ".join(unknown)
        )
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise RerankValidationError("overrides must be an object when provided")
    try:
        validate_option_values(overrides)
    except RerankValidationError as error:
        raise RerankValidationError(f"override {error}") from error
    options = RerankOptions(**{**option_values(config), **overrides})
    validate_rerank_options(options)
    if retrieval:
        workflow = RetrievalOptions(**option_values(payload, RetrievalOptions))
        validation = validate_retrieval_options(workflow, rerank_options=options)
        input_mode = (
            "service"
            if workflow.retriever_host
            else "requests-file"
            if workflow.requests_file
            else "dataset"
        )
    else:
        normalized = normalize_rerank_input(payload)
        validation = {"valid": True, "record_count": 1, "errors": []}
        input_mode = "direct"

    reranker = initialize_reranker(config, options)
    if retrieval:
        # Pointwise/pairwise coordinators retain the previous run's output name.
        # Clear it for each HTTP retrieval request while keeping the model cached.
        coordinator = reranker.get_model_coordinator()
        if hasattr(coordinator, "_filename"):
            coordinator._filename = ""
        results = run_retrieve_and_rerank(
            options=options, retrieval=workflow, reranker=reranker
        )
    else:
        results = run_rerank(options=options, **normalized, reranker=reranker)
    return CommandResponse(
        command="rerank",
        validation=validation,
        inputs={"mode": input_mode, "transport": "http"},
        resolved={
            "model_path": options.model_path,
            "input_mode": input_mode,
            "transport": "http",
        },
        artifacts=[make_data_artifact("rerank-results", serialize_data(results))],
    )


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

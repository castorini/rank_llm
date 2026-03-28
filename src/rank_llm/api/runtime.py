from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from rank_llm.cli.adapters import make_data_artifact, serialize_data
from rank_llm.cli.error_utils import classify_exception
from rank_llm.cli.introspection import validate_rerank_payload
from rank_llm.cli.operations import normalize_direct_rerank_input, run_mcp_rerank
from rank_llm.cli.responses import CommandResponse
from rank_llm.cli.spec import EXIT_CODES
from rank_llm.rerank import Reranker


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8082
    model_path: str = ""
    batch_size: int = 32
    top_k_rerank: int = -1
    context_size: int = 4096
    num_gpus: int = 1
    prompt_template_path: str = ""
    num_few_shot_examples: int = 0
    few_shot_file: str = ""
    shuffle_candidates: bool = False
    print_prompts_responses: bool = False
    use_azure_openai: bool = False
    use_openrouter: bool = False
    base_url: str = ""
    variable_passages: bool = False
    num_passes: int = 1
    window_size: int = 20
    stride: int = 10
    system_message: str = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."
    populate_invocations_history: bool = False
    is_thinking: bool = False
    reasoning_token_budget: int = 10000
    use_logits: bool = False
    use_alpha: bool = False
    sglang_batched: bool = False
    tensorrt_batched: bool = False
    _reranker_cache: dict[tuple[tuple[str, Any], ...], Reranker] = field(
        default_factory=dict, init=False, repr=False
    )


_OVERRIDABLE_FIELDS = {
    "model_path",
    "batch_size",
    "top_k_rerank",
    "context_size",
    "num_gpus",
    "prompt_template_path",
    "num_few_shot_examples",
    "few_shot_file",
    "shuffle_candidates",
    "print_prompts_responses",
    "use_azure_openai",
    "use_openrouter",
    "base_url",
    "variable_passages",
    "num_passes",
    "window_size",
    "stride",
    "system_message",
    "populate_invocations_history",
    "is_thinking",
    "reasoning_token_budget",
    "use_logits",
    "use_alpha",
    "sglang_batched",
    "tensorrt_batched",
}

_RERANKER_CACHE_FIELDS = (
    "model_path",
    "batch_size",
    "context_size",
    "num_gpus",
    "prompt_template_path",
    "num_few_shot_examples",
    "few_shot_file",
    "shuffle_candidates",
    "print_prompts_responses",
    "use_azure_openai",
    "use_openrouter",
    "base_url",
    "variable_passages",
    "num_passes",
    "window_size",
    "stride",
    "system_message",
    "populate_invocations_history",
    "is_thinking",
    "reasoning_token_budget",
    "use_logits",
    "use_alpha",
    "sglang_batched",
    "tensorrt_batched",
)


def _extract_override_payload(payload: dict[str, Any]) -> dict[str, Any]:
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be an object when provided")
    unknown_keys = sorted(set(overrides) - _OVERRIDABLE_FIELDS)
    if unknown_keys:
        raise ValueError(
            "unsupported rerank override field(s): " + ", ".join(unknown_keys)
        )
    return overrides


def _merge_config_with_payload(
    payload: dict[str, Any], *, config: ServerConfig
) -> ServerConfig:
    overrides = _extract_override_payload(payload)
    if not overrides:
        return config
    effective_config = replace(config, **overrides)
    if effective_config.use_azure_openai and effective_config.use_openrouter:
        raise ValueError(
            "use_azure_openai and use_openrouter cannot both be true in overrides"
        )
    return effective_config


def _cache_key(config: ServerConfig) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (field_name, getattr(config, field_name))
        for field_name in _RERANKER_CACHE_FIELDS
    )


def initialize_reranker(
    config: ServerConfig, effective_config: ServerConfig | None = None
) -> Reranker:
    effective_config = effective_config or config
    cache_key = _cache_key(effective_config)
    cached = config._reranker_cache.get(cache_key)
    if cached is not None:
        return cached
    reranker = Reranker(
        Reranker.create_model_coordinator(
            effective_config.model_path,
            None,
            False,
            batch_size=effective_config.batch_size,
            context_size=effective_config.context_size,
            num_gpus=effective_config.num_gpus,
            prompt_template_path=effective_config.prompt_template_path or None,
            num_few_shot_examples=effective_config.num_few_shot_examples,
            few_shot_file=effective_config.few_shot_file or None,
            shuffle_candidates=effective_config.shuffle_candidates,
            print_prompts_responses=effective_config.print_prompts_responses,
            use_azure_openai=effective_config.use_azure_openai,
            use_openrouter=effective_config.use_openrouter,
            base_url=effective_config.base_url or None,
            variable_passages=effective_config.variable_passages,
            num_passes=effective_config.num_passes,
            window_size=effective_config.window_size,
            stride=effective_config.stride,
            system_message=effective_config.system_message,
            populate_invocations_history=effective_config.populate_invocations_history,
            is_thinking=effective_config.is_thinking,
            reasoning_token_budget=effective_config.reasoning_token_budget,
            use_logits=effective_config.use_logits,
            use_alpha=effective_config.use_alpha,
            sglang_batched=effective_config.sglang_batched,
            tensorrt_batched=effective_config.tensorrt_batched,
        )
    )
    config._reranker_cache[cache_key] = reranker
    return reranker


def run_rerank_request(
    payload: dict[str, Any], *, config: ServerConfig
) -> CommandResponse:
    validation = validate_rerank_payload(payload)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))

    normalized = normalize_direct_rerank_input(payload)
    effective_config = _merge_config_with_payload(payload, config=config)
    reranker = initialize_reranker(config, effective_config)
    results = run_mcp_rerank(
        model_path=effective_config.model_path,
        query_text=normalized["query_text"],
        query_id=normalized["query_id"],
        candidates=normalized["candidates"],
        batch_size=effective_config.batch_size,
        top_k_rerank=effective_config.top_k_rerank,
        context_size=effective_config.context_size,
        num_gpus=effective_config.num_gpus,
        prompt_template_path=effective_config.prompt_template_path,
        num_few_shot_examples=effective_config.num_few_shot_examples,
        few_shot_file=effective_config.few_shot_file,
        shuffle_candidates=effective_config.shuffle_candidates,
        print_prompts_responses=effective_config.print_prompts_responses,
        use_azure_openai=effective_config.use_azure_openai,
        use_openrouter=effective_config.use_openrouter,
        base_url=effective_config.base_url,
        variable_passages=effective_config.variable_passages,
        num_passes=effective_config.num_passes,
        window_size=effective_config.window_size,
        stride=effective_config.stride,
        system_message=effective_config.system_message,
        populate_invocations_history=effective_config.populate_invocations_history,
        is_thinking=effective_config.is_thinking,
        reasoning_token_budget=effective_config.reasoning_token_budget,
        use_logits=effective_config.use_logits,
        use_alpha=effective_config.use_alpha,
        sglang_batched=effective_config.sglang_batched,
        tensorrt_batched=effective_config.tensorrt_batched,
        reranker=reranker,
    )
    return CommandResponse(
        command="rerank",
        validation=validation,
        inputs={"mode": "direct", "transport": "http"},
        resolved={
            "model_path": effective_config.model_path,
            "input_mode": "direct",
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

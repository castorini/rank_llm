from __future__ import annotations

from dataclasses import dataclass, field
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
    _reranker: Reranker | None = field(default=None, init=False, repr=False)


def initialize_reranker(config: ServerConfig) -> None:
    if config._reranker is not None:
        return
    config._reranker = Reranker(
        Reranker.create_model_coordinator(
            config.model_path,
            None,
            False,
            batch_size=config.batch_size,
            context_size=config.context_size,
            num_gpus=config.num_gpus,
            prompt_template_path=config.prompt_template_path or None,
            num_few_shot_examples=config.num_few_shot_examples,
            few_shot_file=config.few_shot_file or None,
            shuffle_candidates=config.shuffle_candidates,
            print_prompts_responses=config.print_prompts_responses,
            use_azure_openai=config.use_azure_openai,
            use_openrouter=config.use_openrouter,
            base_url=config.base_url or None,
            variable_passages=config.variable_passages,
            num_passes=config.num_passes,
            window_size=config.window_size,
            stride=config.stride,
            system_message=config.system_message,
            populate_invocations_history=config.populate_invocations_history,
            is_thinking=config.is_thinking,
            reasoning_token_budget=config.reasoning_token_budget,
            use_logits=config.use_logits,
            use_alpha=config.use_alpha,
            sglang_batched=config.sglang_batched,
            tensorrt_batched=config.tensorrt_batched,
        )
    )


def run_rerank_request(
    payload: dict[str, Any], *, config: ServerConfig
) -> CommandResponse:
    validation = validate_rerank_payload(payload)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))

    normalized = normalize_direct_rerank_input(payload)
    initialize_reranker(config)
    results = run_mcp_rerank(
        model_path=config.model_path,
        query_text=normalized["query_text"],
        query_id=normalized["query_id"],
        candidates=normalized["candidates"],
        batch_size=config.batch_size,
        top_k_rerank=config.top_k_rerank,
        context_size=config.context_size,
        num_gpus=config.num_gpus,
        prompt_template_path=config.prompt_template_path,
        num_few_shot_examples=config.num_few_shot_examples,
        few_shot_file=config.few_shot_file,
        shuffle_candidates=config.shuffle_candidates,
        print_prompts_responses=config.print_prompts_responses,
        use_azure_openai=config.use_azure_openai,
        use_openrouter=config.use_openrouter,
        base_url=config.base_url,
        variable_passages=config.variable_passages,
        num_passes=config.num_passes,
        window_size=config.window_size,
        stride=config.stride,
        system_message=config.system_message,
        populate_invocations_history=config.populate_invocations_history,
        is_thinking=config.is_thinking,
        reasoning_token_budget=config.reasoning_token_budget,
        use_logits=config.use_logits,
        use_alpha=config.use_alpha,
        sglang_batched=config.sglang_batched,
        tensorrt_batched=config.tensorrt_batched,
        reranker=config._reranker,
    )
    return CommandResponse(
        command="rerank",
        validation=validation,
        inputs={"mode": "direct", "transport": "http"},
        resolved={
            "model_path": config.model_path,
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

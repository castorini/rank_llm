"""Shared option definitions, argument conversion, schemas, and validation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, get_args, get_type_hints
from urllib.parse import urlsplit

from rank_llm.data import (
    RerankValidationError,
    read_requests_from_file,
)
from rank_llm.retrieve.retrieval_method import RetrievalMethod


@dataclass
class RerankOptions:
    model_path: str = ""
    batch_size: int = field(default=32, metadata={"minimum": 1})
    top_k_rerank: int = field(default=-1, metadata={"minimum": -1})
    context_size: int = field(default=4096, metadata={"minimum": 1})
    num_gpus: int = field(default=1, metadata={"minimum": 0})
    prompt_template_path: str = ""
    num_few_shot_examples: int = field(default=0, metadata={"minimum": 0})
    few_shot_file: str = ""
    shuffle_candidates: bool = False
    print_prompts_responses: bool = False
    use_azure_openai: bool = False
    use_openrouter: bool = False
    use_litellm: bool = False
    base_url: str = ""
    variable_passages: bool = False
    num_passes: int = field(default=1, metadata={"minimum": 1})
    window_size: int = field(default=20, metadata={"minimum": 1})
    stride: int = field(default=10, metadata={"minimum": 1})
    system_message: str = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."
    populate_invocations_history: bool = False
    is_thinking: bool = False
    reasoning_token_budget: int = field(default=10000, metadata={"minimum": 0})
    reasoning_effort: str | None = field(
        default=None,
        metadata={"enum": ["none", "minimal", "low", "medium", "high", "xhigh", None]},
    )
    use_logits: bool = False
    use_alpha: bool = False
    pointwise_vllm: bool = False
    listwise_vllm_with_openai_sdk: bool = False
    max_passage_words: int = field(default=300, metadata={"minimum": 1})

    def to_kwargs(self) -> dict[str, Any]:
        """Translate optional CLI strings to the pipeline's None defaults."""
        values = option_values(self)
        for name in ("prompt_template_path", "few_shot_file", "base_url"):
            values[name] = values[name] or None
        return values


@dataclass
class RetrievalOptions:
    query: str = ""
    query_id: str | int = 1
    dataset: str = ""
    requests_file: str = ""
    retriever_host: str = ""
    retrieval_method: str = field(
        default="unspecified", metadata={"enum": [m.value for m in RetrievalMethod]}
    )
    top_k_candidates: int = field(default=100, metadata={"minimum": 1})
    max_queries: int = field(default=-1, metadata={"minimum": -1})
    qrels_file: str = ""
    output_jsonl_file: str = ""
    output_trec_file: str = ""
    invocations_history_file: str = ""


def option_values(source: Any, option_type: type = RerankOptions) -> dict[str, Any]:
    """Select options from a namespace/config, supplying canonical defaults."""
    values = source if isinstance(source, dict) else vars(source)
    return {f.name: values.get(f.name, f.default) for f in fields(option_type)}


def option_schema(option_type: type = RerankOptions) -> dict[str, Any]:
    hints = get_type_hints(option_type)
    names = {str: "string", int: "integer", bool: "boolean", type(None): "null"}
    properties = {}
    for f in fields(option_type):
        types = get_args(hints[f.name]) or (hints[f.name],)
        json_types = [names[t] for t in types]
        properties[f.name] = {
            "type": json_types[0] if len(json_types) == 1 else json_types,
            "default": f.default,
            **dict(f.metadata),
        }
    return {"type": "object", "properties": properties, "additionalProperties": False}


def add_option_arguments(parser: Any, option_type: type = RerankOptions) -> None:
    """Register CLI flags from the same definitions used by server schemas."""
    hints = get_type_hints(option_type)
    for f in fields(option_type):
        kwargs: dict[str, Any] = {"dest": f.name, "default": f.default}
        types = get_args(hints[f.name]) or (hints[f.name],)
        if types == (bool,):
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = types[0]
        if "enum" in f.metadata:
            kwargs["choices"] = [v for v in f.metadata["enum"] if v is not None]
        if f.name == "model_path":
            kwargs["required"] = True
        parser.add_argument("--" + f.name.replace("_", "-"), **kwargs)


def validate_option_values(
    values: dict[str, Any], option_type: type = RerankOptions
) -> None:
    schema = option_schema(option_type)["properties"]
    unknown = sorted(set(values) - set(schema))
    if unknown:
        raise RerankValidationError(
            "unsupported rerank field(s): " + ", ".join(unknown)
        )
    python_types = {"string": str, "integer": int, "boolean": bool, "null": type(None)}
    for name, value in values.items():
        definition = schema[name]
        kinds = definition["type"]
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not any(type(value) is python_types[kind] for kind in kinds):
            article = "an" if kinds[0] == "integer" else "a"
            raise RerankValidationError(
                f"'{name}' must be {article} {' or '.join(kinds)}"
            )
        if "enum" in definition and value not in definition["enum"]:
            raise RerankValidationError(
                f"'{name}' must be one of: {', '.join(sorted(str(v) for v in definition['enum'] if v is not None))}"
            )
        if "minimum" in definition and value < definition["minimum"]:
            raise RerankValidationError(
                f"'{name}' must be at least {definition['minimum']}"
            )


def validate_rerank_options(options: RerankOptions) -> None:
    values = option_values(options)
    validate_option_values(values)
    if not values["model_path"].strip():
        raise RerankValidationError("model_path is required")
    for name in ("pointwise_vllm", "listwise_vllm_with_openai_sdk"):
        if values[name] and not values["base_url"].strip():
            flag = name.replace("_", "-")
            raise RerankValidationError(f"--base-url is required when --{flag} is set")
    enabled = [
        name
        for name in (
            "use_azure_openai",
            "use_openrouter",
            "use_litellm",
            "pointwise_vllm",
            "listwise_vllm_with_openai_sdk",
        )
        if values[name]
    ]
    if len(enabled) > 1:
        raise RerankValidationError(
            "backend selectors cannot be combined: " + ", ".join(enabled)
        )


def validate_retrieval_options(
    options: RetrievalOptions, *, rerank_options: RerankOptions, check_file: bool = True
) -> dict[str, Any]:
    values = option_values(options, RetrievalOptions)
    method = values["retrieval_method"]
    values["retrieval_method"] = (
        method.value if isinstance(method, RetrievalMethod) else method
    )
    validate_option_values(values, RetrievalOptions)
    if bool(values["dataset"]) == bool(values["requests_file"]):
        raise RerankValidationError("provide exactly one of dataset or requests_file")
    if values["requests_file"] and values["retrieval_method"] != "unspecified":
        raise RerankValidationError(
            "--retrieval-method must not be used with --requests-file"
        )
    if values["dataset"] and values["retrieval_method"] == "unspecified":
        raise RerankValidationError(
            "--retrieval-method is required when --dataset is provided"
        )
    if values["requests_file"] and values["query"]:
        raise RerankValidationError("query must not be used with requests_file")
    if values["retriever_host"]:
        host = urlsplit(values["retriever_host"])
        if (
            host.scheme not in ("http", "https")
            or not host.netloc
            or host.query
            or host.fragment
        ):
            raise RerankValidationError("retriever_host must be an HTTP(S) base URL")
        if (
            not values["dataset"]
            or not values["query"].strip()
            or values["retrieval_method"] != "bm25"
        ):
            raise RerankValidationError(
                "service retrieval requires dataset, a nonempty query, and BM25"
            )
    if (
        values["requests_file"]
        and rerank_options.populate_invocations_history
        and not values["invocations_history_file"]
    ):
        raise RerankValidationError(
            "invocations_history_file is required when populating request-file history"
        )
    count = (
        len(read_requests_from_file(values["requests_file"]))
        if check_file and values["requests_file"]
        else 0
    )
    return {"valid": True, "record_count": count, "errors": []}

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any, NoReturn

from rank_llm.cli.adapters import make_data_artifact, serialize_data
from rank_llm.cli.config import load_config
from rank_llm.cli.error_utils import classify_exception, has_partial_success_metrics
from rank_llm.cli.introspection import (
    COMMAND_DESCRIPTIONS,
    SCHEMAS,
    doctor_report,
    validate_rerank_batch_file,
    validate_rerank_payload,
)
from rank_llm.cli.operations import (
    normalize_direct_rerank_input,
    run_evaluate_aggregate,
    run_mcp_rerank,
    run_mcp_retrieve_and_rerank,
    run_response_analysis_files,
    run_retrieve_cache_generation,
)
from rank_llm.cli.prompt_view import (
    PromptTemplateError,
    build_prompt_template_view,
    build_rendered_prompt_view,
    list_prompt_templates,
    render_prompt_catalog_text,
    render_prompt_template_text,
    render_rendered_prompt_text,
)
from rank_llm.cli.responses import CommandResponse
from rank_llm.cli.spec import EXIT_CODES, KNOWN_COMMANDS, TOP_LEVEL_EXAMPLES
from rank_llm.cli.view import ViewError, build_view_summary, render_view_summary
from rank_llm.retrieve.retrieval_method import RetrievalMethod


class CLIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        exit_code: int,
        status: str,
        error_code: str,
        command: str = "unknown",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        self.status = status
        self.error_code = error_code
        self.command = command
        self.details = details or {}


class CLIArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._current_argv: list[str] = []

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        self._current_argv = list(args) if args is not None else list(sys.argv[1:])
        return super().parse_args(args, namespace)

    def error(self, message: str) -> NoReturn:
        if message == "the following arguments are required: command":
            raise CLIError(
                _build_missing_command_message(),
                exit_code=EXIT_CODES["invalid_arguments"],
                status="validation_error",
                error_code="missing_command",
                details={
                    "available_commands": list(KNOWN_COMMANDS),
                    "examples": list(TOP_LEVEL_EXAMPLES),
                    "help_hint": "Run `rank-llm --help` for full usage.",
                },
            )
        raise CLIError(
            message,
            exit_code=EXIT_CODES["invalid_arguments"],
            status="validation_error",
            error_code="invalid_arguments",
            command=_detect_command(self._current_argv),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = CLIArgumentParser(
        prog="rank-llm",
        description="Packaged CLI entrypoint for RankLLM.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Render command output as plain text or JSON envelope.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    rerank_parser = subparsers.add_parser("rerank", help="Run RankLLM reranking.")
    rerank_parser.add_argument("--model-path", required=True, dest="model_path")
    rerank_parser.add_argument("--query", default="")
    rerank_parser.add_argument("--dataset")
    rerank_parser.add_argument("--requests-file", dest="requests_file")
    rerank_parser.add_argument("--input-json", dest="input_json")
    rerank_parser.add_argument("--stdin", action="store_true")
    rerank_parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    rerank_parser.add_argument(
        "--validate-only",
        dest="validate_only",
        action="store_true",
    )
    rerank_parser.add_argument(
        "--retrieval-method",
        dest="retrieval_method",
        type=RetrievalMethod,
        choices=list(RetrievalMethod),
    )
    rerank_parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    rerank_parser.add_argument(
        "--top-k-candidates", dest="top_k_candidates", type=int, default=100
    )
    rerank_parser.add_argument(
        "--top-k-rerank", dest="top_k_rerank", type=int, default=-1
    )
    rerank_parser.add_argument(
        "--max-queries", dest="max_queries", type=int, default=-1
    )
    rerank_parser.add_argument(
        "--context-size", dest="context_size", type=int, default=4096
    )
    rerank_parser.add_argument("--num-gpus", dest="num_gpus", type=int, default=1)
    rerank_parser.add_argument(
        "--prompt-template-path", dest="prompt_template_path", default=""
    )
    rerank_parser.add_argument(
        "--num-few-shot-examples", dest="num_few_shot_examples", type=int, default=0
    )
    rerank_parser.add_argument("--few-shot-file", dest="few_shot_file", default="")
    rerank_parser.add_argument("--qrels-file", dest="qrels_file", default="")
    rerank_parser.add_argument(
        "--output-jsonl-file", dest="output_jsonl_file", default=""
    )
    rerank_parser.add_argument(
        "--output-trec-file", dest="output_trec_file", default=""
    )
    rerank_parser.add_argument(
        "--invocations-history-file", dest="invocations_history_file", default=""
    )
    rerank_parser.add_argument(
        "--shuffle-candidates", dest="shuffle_candidates", action="store_true"
    )
    rerank_parser.add_argument(
        "--print-prompts-responses", dest="print_prompts_responses", action="store_true"
    )
    rerank_parser.add_argument(
        "--use-azure-openai", dest="use_azure_openai", action="store_true"
    )
    rerank_parser.add_argument(
        "--use-openrouter", dest="use_openrouter", action="store_true"
    )
    rerank_parser.add_argument("--base-url", dest="base_url", default="")
    rerank_parser.add_argument(
        "--variable-passages", dest="variable_passages", action="store_true"
    )
    rerank_parser.add_argument("--num-passes", dest="num_passes", type=int, default=1)
    rerank_parser.add_argument(
        "--window-size", dest="window_size", type=int, default=20
    )
    rerank_parser.add_argument("--stride", dest="stride", type=int, default=10)
    rerank_parser.add_argument(
        "--system-message",
        dest="system_message",
        default="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
    )
    rerank_parser.add_argument(
        "--populate-invocations-history",
        dest="populate_invocations_history",
        action="store_true",
    )
    rerank_parser.add_argument("--is-thinking", dest="is_thinking", action="store_true")
    rerank_parser.add_argument(
        "--reasoning-token-budget",
        dest="reasoning_token_budget",
        type=int,
        default=10000,
    )
    rerank_parser.add_argument("--use-logits", dest="use_logits", action="store_true")
    rerank_parser.add_argument("--use-alpha", dest="use_alpha", action="store_true")
    rerank_parser.add_argument(
        "--sglang-batched", dest="sglang_batched", action="store_true"
    )
    rerank_parser.add_argument(
        "--tensorrt-batched", dest="tensorrt_batched", action="store_true"
    )
    rerank_parser.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        choices=("low", "medium", "high"),
        default=None,
    )
    rerank_parser.add_argument(
        "--max-passage-words",
        dest="max_passage_words",
        type=int,
        default=300,
    )
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate inputs without executing a model.",
    )
    validate_subparsers = validate_parser.add_subparsers(
        dest="validate_target",
        required=True,
    )
    validate_rerank_parser = validate_subparsers.add_parser(
        "rerank",
        help="Validate rerank inputs without executing models.",
    )
    validate_rerank_parser.add_argument("--input-json", dest="input_json")
    validate_rerank_parser.add_argument("--stdin", action="store_true")
    validate_rerank_parser.add_argument("--requests-file", dest="requests_file")

    prompt_parser = subparsers.add_parser(
        "prompt",
        help="Inspect bundled prompt templates.",
    )
    prompt_subparsers = prompt_parser.add_subparsers(
        dest="prompt_command",
        required=True,
    )
    prompt_subparsers.add_parser("list", help="List bundled prompt templates.")
    prompt_show_parser = prompt_subparsers.add_parser(
        "show",
        help="Show a bundled or custom prompt template.",
    )
    prompt_show_parser.add_argument("name")
    prompt_render_parser = prompt_subparsers.add_parser(
        "render",
        help="Render a prompt template with direct input payload.",
    )
    prompt_render_parser.add_argument("name")
    prompt_render_parser.add_argument("--input-json", dest="input_json")
    prompt_render_parser.add_argument("--stdin", action="store_true")

    view_parser = subparsers.add_parser(
        "view",
        help="Inspect RankLLM artifacts and outputs.",
    )
    view_parser.add_argument("path")
    view_parser.add_argument("--records", type=int, default=1)

    describe_parser = subparsers.add_parser(
        "describe",
        help="Show structured metadata for a CLI command.",
    )
    describe_parser.add_argument("name", choices=sorted(COMMAND_DESCRIPTIONS))

    schema_parser = subparsers.add_parser(
        "schema",
        help="Show JSON schemas for supported contracts.",
    )
    schema_parser.add_argument("name", choices=sorted(SCHEMAS))

    subparsers.add_parser(
        "doctor",
        help="Report environment and dependency readiness.",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start RankLLM transport servers.",
    )
    serve_subparsers = serve_parser.add_subparsers(
        dest="serve_target",
        required=True,
    )
    serve_http_parser = serve_subparsers.add_parser(
        "http",
        help="Start the RankLLM HTTP server.",
        description="Start the RankLLM HTTP server.",
    )
    serve_http_parser.add_argument("--host", default="0.0.0.0")
    serve_http_parser.add_argument("--port", type=int, default=8082)
    serve_http_parser.add_argument("--model-path", required=True, dest="model_path")
    serve_http_parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=32
    )
    serve_http_parser.add_argument(
        "--top-k-rerank",
        dest="top_k_rerank",
        type=int,
        default=-1,
    )
    serve_http_parser.add_argument(
        "--context-size",
        dest="context_size",
        type=int,
        default=4096,
    )
    serve_http_parser.add_argument("--num-gpus", dest="num_gpus", type=int, default=1)
    serve_http_parser.add_argument(
        "--prompt-template-path",
        dest="prompt_template_path",
        default="",
    )
    serve_http_parser.add_argument(
        "--num-few-shot-examples",
        dest="num_few_shot_examples",
        type=int,
        default=0,
    )
    serve_http_parser.add_argument("--few-shot-file", dest="few_shot_file", default="")
    serve_http_parser.add_argument(
        "--shuffle-candidates",
        dest="shuffle_candidates",
        action="store_true",
    )
    serve_http_parser.add_argument(
        "--print-prompts-responses",
        dest="print_prompts_responses",
        action="store_true",
    )
    serve_http_parser.add_argument(
        "--use-azure-openai",
        dest="use_azure_openai",
        action="store_true",
    )
    serve_http_parser.add_argument(
        "--use-openrouter",
        dest="use_openrouter",
        action="store_true",
    )
    serve_http_parser.add_argument("--base-url", dest="base_url", default="")
    serve_http_parser.add_argument(
        "--variable-passages",
        dest="variable_passages",
        action="store_true",
    )
    serve_http_parser.add_argument(
        "--num-passes", dest="num_passes", type=int, default=1
    )
    serve_http_parser.add_argument(
        "--window-size", dest="window_size", type=int, default=20
    )
    serve_http_parser.add_argument("--stride", dest="stride", type=int, default=10)
    serve_http_parser.add_argument(
        "--system-message",
        dest="system_message",
        default="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
    )
    serve_http_parser.add_argument(
        "--populate-invocations-history",
        dest="populate_invocations_history",
        action="store_true",
    )
    serve_http_parser.add_argument(
        "--is-thinking",
        dest="is_thinking",
        action="store_true",
    )
    serve_http_parser.add_argument(
        "--reasoning-token-budget",
        dest="reasoning_token_budget",
        type=int,
        default=10000,
    )
    serve_http_parser.add_argument(
        "--use-logits",
        dest="use_logits",
        action="store_true",
    )
    serve_http_parser.add_argument(
        "--use-alpha",
        dest="use_alpha",
        action="store_true",
    )
    serve_http_parser.add_argument(
        "--sglang-batched",
        dest="sglang_batched",
        action="store_true",
    )
    serve_http_parser.add_argument(
        "--tensorrt-batched",
        dest="tensorrt_batched",
        action="store_true",
    )
    serve_mcp_parser = serve_subparsers.add_parser(
        "mcp",
        help="Start the RankLLM MCP server.",
        description="Start the RankLLM MCP server.",
    )
    serve_mcp_parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
    )
    serve_mcp_parser.add_argument("--port", type=int, default=8000)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Aggregate trec_eval metrics across rerank outputs.",
    )
    evaluate_parser.add_argument("--model-name", required=True, dest="model_name")
    evaluate_parser.add_argument("--context-size", type=int, default=4096)
    evaluate_parser.add_argument(
        "--rerank-results-dirname",
        dest="rerank_results_dirname",
        default="rerank_results",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze stored RankLLM responses.",
    )
    analyze_parser.add_argument("--files", nargs="+", required=True)
    analyze_parser.add_argument("--verbose", action="store_true")

    retrieve_cache_parser = subparsers.add_parser(
        "retrieve-cache",
        help="Generate cached retrieval JSON from run files.",
    )
    retrieve_cache_parser.add_argument("--trec-file", required=True, dest="trec_file")
    retrieve_cache_parser.add_argument(
        "--collection-file",
        required=True,
        dest="collection_file",
    )
    retrieve_cache_parser.add_argument("--query-file", required=True, dest="query_file")
    retrieve_cache_parser.add_argument(
        "--output-file",
        required=True,
        dest="output_file",
    )
    retrieve_cache_parser.add_argument(
        "--output-trec-file",
        dest="output_trec_file",
        default=None,
    )
    retrieve_cache_parser.add_argument("--topk", type=int, default=20)

    for command in KNOWN_COMMANDS:
        if command == "rerank":
            continue
        if command == "validate":
            continue
        if command == "prompt":
            continue
        if command == "view":
            continue
        if command == "describe":
            continue
        if command == "schema":
            continue
        if command == "doctor":
            continue
        if command in {"evaluate", "analyze", "retrieve-cache", "serve"}:
            continue
        subparsers.add_parser(command, help=argparse.SUPPRESS)
    return parser


def _detect_command(argv: Sequence[str]) -> str:
    for token in argv:
        if token in KNOWN_COMMANDS:
            return token
    return "unknown"


def _build_missing_command_message() -> str:
    command_list = ", ".join(KNOWN_COMMANDS)
    examples = "\n".join(f"  {example}" for example in TOP_LEVEL_EXAMPLES)
    return (
        "No command provided. Choose one of: "
        f"{command_list}\n"
        "Examples:\n"
        f"{examples}\n"
        "Run `rank-llm --help` for full usage."
    )


def _wants_json(argv: Sequence[str]) -> bool:
    for index, token in enumerate(argv):
        if token == "--output" and index + 1 < len(argv):
            return argv[index + 1] == "json"
        if token == "--output=json":
            return True
    return False


def _emit_json(data: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(data) + "\n")


def _build_error_response(error: CLIError) -> CommandResponse:
    return CommandResponse(
        command=error.command,
        status=error.status,
        exit_code=error.exit_code,
        errors=[
            {
                "code": error.error_code,
                "message": error.message,
                "details": error.details,
                "retryable": False,
            }
        ],
    )


def _read_direct_payload(args: argparse.Namespace) -> dict[str, Any]:
    try:
        if args.stdin:
            return json.loads(sys.stdin.read())
        if args.input_json:
            return json.loads(args.input_json)
    except json.JSONDecodeError as exc:
        source = "stdin" if args.stdin else "--input-json"
        raise CLIError(
            f"Invalid JSON payload provided via {source}: {exc.msg}",
            exit_code=EXIT_CODES["invalid_arguments"],
            status="validation_error",
            error_code="invalid_json",
            command="rerank",
            details={"source": source, "line": exc.lineno, "column": exc.colno},
        ) from exc
    raise CLIError(
        "Direct input requires --stdin or --input-json",
        exit_code=EXIT_CODES["invalid_arguments"],
        status="validation_error",
        error_code="missing_direct_input",
        command="rerank",
    )


def _validate_rerank_sources(args: argparse.Namespace) -> None:
    if args.dataset or args.requests_file or args.input_json is not None or args.stdin:
        return
    raise CLIError(
        "Rerank requires one input source: --dataset, --requests-file, --input-json, or --stdin",
        exit_code=EXIT_CODES["invalid_arguments"],
        status="validation_error",
        error_code="missing_input_source",
        command="rerank",
    )


def _validate_rerank_execution_args(args: argparse.Namespace) -> None:
    if args.requests_file and args.retrieval_method:
        raise CLIError(
            "--retrieval-method must not be used with --requests-file",
            exit_code=EXIT_CODES["invalid_arguments"],
            status="validation_error",
            error_code="invalid_arguments",
            command="rerank",
        )
    if args.dataset and not args.retrieval_method:
        raise CLIError(
            "--retrieval-method is required when --dataset is provided",
            exit_code=EXIT_CODES["invalid_arguments"],
            status="validation_error",
            error_code="invalid_arguments",
            command="rerank",
        )


def _normalize_direct_rerank_input(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload["query"]
    query_text = query["text"] if isinstance(query, dict) else query
    query_id = query.get("qid", "") if isinstance(query, dict) else ""
    candidates = []
    for index, candidate in enumerate(payload["candidates"], start=1):
        if isinstance(candidate, str):
            candidates.append({"docid": str(index), "score": 0.0, "doc": candidate})
            continue
        if "text" in candidate:
            candidates.append(
                {
                    "docid": candidate.get("docid", str(index)),
                    "score": candidate.get("score", 0.0),
                    "doc": candidate["text"],
                }
            )
            continue
        candidates.append(
            {
                "docid": candidate.get("docid", str(index)),
                "score": candidate.get("score", 0.0),
                "doc": candidate["doc"],
            }
        )
    return {"query_text": query_text, "query_id": query_id, "candidates": candidates}


def _validation_error_response(
    command: str,
    validation: dict[str, Any],
) -> CommandResponse:
    return CommandResponse(
        command=command,
        status="validation_error",
        exit_code=EXIT_CODES["validation_error"],
        validation=validation,
        errors=[
            {
                "code": "validation_error",
                "message": "; ".join(validation.get("errors", ["validation failed"])),
                "details": validation,
                "retryable": False,
            }
        ],
    )


def _run_rerank_command(args: argparse.Namespace) -> CommandResponse:
    _validate_rerank_sources(args)
    direct_mode = args.input_json is not None or args.stdin
    if direct_mode:
        payload = _read_direct_payload(args)
        validation = validate_rerank_payload(payload)
        if not validation["valid"]:
            return _validation_error_response("rerank", validation)
        if args.validate_only or args.dry_run:
            return CommandResponse(
                command="rerank",
                mode="validate" if args.validate_only else "dry_run",
                validation=validation,
                inputs={"mode": "direct"},
                resolved={"model_path": args.model_path, "input_mode": "direct"},
            )
        normalized = normalize_direct_rerank_input(payload)
        results = run_mcp_rerank(
            model_path=args.model_path,
            query_text=normalized["query_text"],
            query_id=normalized["query_id"],
            candidates=normalized["candidates"],
            batch_size=args.batch_size,
            top_k_rerank=args.top_k_rerank,
            context_size=args.context_size,
            num_gpus=args.num_gpus,
            prompt_template_path=args.prompt_template_path,
            num_few_shot_examples=args.num_few_shot_examples,
            few_shot_file=args.few_shot_file,
            shuffle_candidates=args.shuffle_candidates,
            print_prompts_responses=args.print_prompts_responses,
            use_azure_openai=args.use_azure_openai,
            use_openrouter=args.use_openrouter,
            base_url=args.base_url,
            variable_passages=args.variable_passages,
            num_passes=args.num_passes,
            window_size=args.window_size,
            stride=args.stride,
            system_message=args.system_message,
            populate_invocations_history=args.populate_invocations_history,
            is_thinking=args.is_thinking,
            reasoning_token_budget=args.reasoning_token_budget,
            use_logits=args.use_logits,
            use_alpha=args.use_alpha,
            sglang_batched=args.sglang_batched,
            tensorrt_batched=args.tensorrt_batched,
            reasoning_effort=args.reasoning_effort,
            max_passage_words=args.max_passage_words,
        )
        input_mode = "direct"
    else:
        validation = {"valid": True, "record_count": 0, "errors": []}
        _validate_rerank_execution_args(args)
        if args.requests_file:
            validation = validate_rerank_batch_file(args.requests_file)
            if not validation["valid"]:
                return _validation_error_response("rerank", validation)
        if args.validate_only or args.dry_run:
            input_mode = "requests-file" if args.requests_file else "dataset"
            return CommandResponse(
                command="rerank",
                mode="validate" if args.validate_only else "dry_run",
                validation=validation,
                inputs={"mode": input_mode},
                resolved={"model_path": args.model_path, "input_mode": input_mode},
            )
        results = run_mcp_retrieve_and_rerank(
            model_path=args.model_path,
            query=args.query,
            batch_size=args.batch_size,
            dataset=args.dataset or "",
            requests_file=args.requests_file or "",
            qrels_file=args.qrels_file,
            output_jsonl_file=args.output_jsonl_file,
            output_trec_file=args.output_trec_file,
            invocations_history_file=args.invocations_history_file,
            retrieval_method=args.retrieval_method or RetrievalMethod.UNSPECIFIED,
            top_k_candidates=args.top_k_candidates,
            top_k_rerank=args.top_k_rerank,
            max_queries=args.max_queries,
            context_size=args.context_size,
            num_gpus=args.num_gpus,
            prompt_template_path=args.prompt_template_path,
            num_few_shot_examples=args.num_few_shot_examples,
            few_shot_file=args.few_shot_file,
            shuffle_candidates=args.shuffle_candidates,
            print_prompts_responses=args.print_prompts_responses,
            use_azure_openai=args.use_azure_openai,
            use_openrouter=args.use_openrouter,
            base_url=args.base_url,
            variable_passages=args.variable_passages,
            num_passes=args.num_passes,
            window_size=args.window_size,
            stride=args.stride,
            system_message=args.system_message,
            populate_invocations_history=args.populate_invocations_history,
            is_thinking=args.is_thinking,
            reasoning_token_budget=args.reasoning_token_budget,
            use_logits=args.use_logits,
            use_alpha=args.use_alpha,
            sglang_batched=args.sglang_batched,
            tensorrt_batched=args.tensorrt_batched,
            reasoning_effort=args.reasoning_effort,
            max_passage_words=args.max_passage_words,
        )
        input_mode = "requests-file" if args.requests_file else "dataset"

    return CommandResponse(
        command="rerank",
        validation={"valid": True, "record_count": 1 if direct_mode else 0},
        inputs={"mode": input_mode},
        resolved={"model_path": args.model_path, "input_mode": input_mode},
        artifacts=[make_data_artifact("rerank-results", serialize_data(results))],
    )


def _run_validate_command(args: argparse.Namespace) -> CommandResponse:
    if args.validate_target != "rerank":
        return CommandResponse(
            command="validate",
            warnings=["validate target not implemented yet."],
        )
    if args.requests_file:
        validation = validate_rerank_batch_file(args.requests_file)
    else:
        validation = validate_rerank_payload(_read_direct_payload(args))
    if not validation["valid"]:
        return _validation_error_response("validate", validation)
    return CommandResponse(
        command="validate",
        mode="validate",
        validation=validation,
        inputs={"target": "rerank"},
        resolved={"target": "rerank"},
    )


def _run_prompt_command(args: argparse.Namespace) -> CommandResponse:
    try:
        if args.prompt_command == "list":
            catalog = list_prompt_templates()
            return CommandResponse(
                command="prompt",
                inputs={"subcommand": "list"},
                artifacts=[make_data_artifact("prompt-catalog", catalog)],
            )
        if args.prompt_command == "show":
            view = build_prompt_template_view(args.name)
            return CommandResponse(
                command="prompt",
                inputs={"subcommand": "show", "name": args.name},
                artifacts=[make_data_artifact("prompt-template", view)],
            )
        if args.prompt_command == "render":
            payload = _read_direct_payload(args)
            validation = validate_rerank_payload(payload)
            if not validation["valid"]:
                return _validation_error_response("prompt", validation)
            view = build_rendered_prompt_view(args.name, payload)
            return CommandResponse(
                command="prompt",
                inputs={"subcommand": "render", "name": args.name},
                validation=validation,
                artifacts=[make_data_artifact("rendered-prompt", view)],
            )
    except PromptTemplateError as exc:
        raise CLIError(
            str(exc),
            exit_code=EXIT_CODES["invalid_arguments"],
            status="validation_error",
            error_code="missing_resource",
            command="prompt",
        ) from exc
    return CommandResponse(
        command="prompt", warnings=["prompt command not implemented yet."]
    )


def _run_view_command(args: argparse.Namespace) -> CommandResponse:
    try:
        summary = build_view_summary(args.path, records=args.records)
    except json.JSONDecodeError as error:
        return _validation_error_response(
            "view",
            {
                "valid": False,
                "record_count": 0,
                "errors": [f"invalid JSON content: {error.msg}"],
            },
        )
    except ViewError as error:
        return _validation_error_response(
            "view",
            {"valid": False, "record_count": 0, "errors": [str(error)]},
        )
    return CommandResponse(
        command="view",
        inputs={"path": args.path},
        resolved={"path": args.path},
        artifacts=[make_data_artifact("view-summary", summary)],
    )


def _run_describe_command(args: argparse.Namespace) -> CommandResponse:
    return CommandResponse(
        command="describe",
        inputs={"name": args.name},
        artifacts=[
            make_data_artifact(
                "command-description",
                {"name": args.name, **COMMAND_DESCRIPTIONS[args.name]},
            )
        ],
    )


def _run_schema_command(args: argparse.Namespace) -> CommandResponse:
    return CommandResponse(
        command="schema",
        inputs={"name": args.name},
        artifacts=[
            make_data_artifact(
                "schema", {"name": args.name, "schema": SCHEMAS[args.name]}
            )
        ],
    )


def _run_doctor_command() -> CommandResponse:
    config, config_path = load_config()
    report = doctor_report()
    report["config_file"] = str(config_path) if config_path else None
    return CommandResponse(
        command="doctor",
        resolved={"config": config},
        artifacts=[make_data_artifact("doctor-output", report)],
    )


def _run_evaluate_command(args: argparse.Namespace) -> CommandResponse:
    summary = run_evaluate_aggregate(
        model_name=args.model_name,
        context_size=args.context_size,
        rerank_results_dirname=args.rerank_results_dirname,
        capture_stdout=args.output == "json",
    )
    return CommandResponse(
        command="evaluate",
        inputs={
            "model_name": args.model_name,
            "context_size": args.context_size,
            "rerank_results_dirname": args.rerank_results_dirname,
        },
        artifacts=[make_data_artifact("evaluation-summary", summary)],
    )


def _run_analyze_command(args: argparse.Namespace) -> CommandResponse:
    summary = run_response_analysis_files(
        files=args.files,
        verbose=args.verbose,
        capture_stdout=args.output == "json",
    )
    status = "success"
    exit_code = EXIT_CODES["success"]
    warnings: list[str] = []
    if has_partial_success_metrics(summary.get("metrics")):
        status = "partial_success"
        exit_code = EXIT_CODES["partial_success"]
        warnings.append(
            "Analyzed responses include a mix of valid outputs and malformed outputs."
        )
    return CommandResponse(
        command="analyze",
        status=status,
        exit_code=exit_code,
        inputs={"files": args.files, "verbose": args.verbose},
        artifacts=[make_data_artifact("analysis-summary", summary)],
        warnings=warnings,
    )


def _run_retrieve_cache_command(args: argparse.Namespace) -> CommandResponse:
    summary = run_retrieve_cache_generation(
        trec_file=args.trec_file,
        collection_file=args.collection_file,
        query_file=args.query_file,
        output_file=args.output_file,
        output_trec_file=args.output_trec_file,
        topk=args.topk,
        capture_stdout=args.output == "json",
    )
    return CommandResponse(
        command="retrieve-cache",
        inputs={
            "trec_file": args.trec_file,
            "collection_file": args.collection_file,
            "query_file": args.query_file,
            "output_file": args.output_file,
            "output_trec_file": args.output_trec_file,
            "topk": args.topk,
        },
        artifacts=[make_data_artifact("retrieve-cache-summary", summary)],
    )


def _run_serve_command(args: argparse.Namespace) -> CommandResponse:
    if args.serve_target == "mcp":
        try:
            from rank_llm.server.mcp.mcp_rankllm import run_mcp_server

            run_mcp_server(transport=args.transport, port=args.port)
        except ImportError as error:
            raise CLIError(
                "serve mcp requires MCP dependencies; install the `mcp` extra",
                exit_code=EXIT_CODES["missing_resource"],
                status="validation_error",
                error_code="missing_mcp_dependencies",
                command="serve",
                details={"missing_dependencies": ["fastmcp", "pyserini"]},
            ) from error

        return CommandResponse(
            command="serve",
            resolved={"target": "mcp", "transport": args.transport, "port": args.port},
        )

    try:
        import uvicorn

        from rank_llm.api.app import create_app
        from rank_llm.api.runtime import ServerConfig
    except ModuleNotFoundError as error:
        raise CLIError(
            "serve http requires FastAPI dependencies; install the `api` extra",
            exit_code=EXIT_CODES["missing_resource"],
            status="validation_error",
            error_code="missing_api_dependencies",
            command="serve",
            details={"missing_dependencies": ["fastapi", "uvicorn"]},
        ) from error

    app = create_app(
        ServerConfig(
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            batch_size=args.batch_size,
            top_k_rerank=args.top_k_rerank,
            context_size=args.context_size,
            num_gpus=args.num_gpus,
            prompt_template_path=args.prompt_template_path,
            num_few_shot_examples=args.num_few_shot_examples,
            few_shot_file=args.few_shot_file,
            shuffle_candidates=args.shuffle_candidates,
            print_prompts_responses=args.print_prompts_responses,
            use_azure_openai=args.use_azure_openai,
            use_openrouter=args.use_openrouter,
            base_url=args.base_url,
            variable_passages=args.variable_passages,
            num_passes=args.num_passes,
            window_size=args.window_size,
            stride=args.stride,
            system_message=args.system_message,
            populate_invocations_history=args.populate_invocations_history,
            is_thinking=args.is_thinking,
            reasoning_token_budget=args.reasoning_token_budget,
            use_logits=args.use_logits,
            use_alpha=args.use_alpha,
            sglang_batched=args.sglang_batched,
            tensorrt_batched=args.tensorrt_batched,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return CommandResponse(
        command="serve",
        resolved={"target": "http", "host": args.host, "port": args.port},
    )


def _run_command(args: argparse.Namespace) -> CommandResponse:
    if args.command == "rerank":
        return _run_rerank_command(args)
    if args.command == "validate":
        return _run_validate_command(args)
    if args.command == "prompt":
        return _run_prompt_command(args)
    if args.command == "view":
        return _run_view_command(args)
    if args.command == "describe":
        return _run_describe_command(args)
    if args.command == "schema":
        return _run_schema_command(args)
    if args.command == "doctor":
        return _run_doctor_command()
    if args.command == "evaluate":
        return _run_evaluate_command(args)
    if args.command == "analyze":
        return _run_analyze_command(args)
    if args.command == "retrieve-cache":
        return _run_retrieve_cache_command(args)
    if args.command == "serve":
        return _run_serve_command(args)
    return CommandResponse(
        command=args.command,
        status="success",
        resolved={"config": load_config()[0]},
        warnings=[f"{args.command} is not implemented yet."],
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    config, config_path = load_config()
    try:
        args = parser.parse_args(argv)
        args._config_path = config_path
        for key, value in config.items():
            flag = f"--{key.replace('_', '-')}"
            if not any(arg == flag or arg.startswith(f"{flag}=") for arg in argv):
                setattr(args, key, value)
        response = _run_command(args)
    except CLIError as error:
        response = _build_error_response(error)
        if _wants_json(argv):
            _emit_json(response.to_envelope())
        else:
            sys.stderr.write(f"{error.message}\n")
        return error.exit_code
    except Exception as error:  # noqa: BLE001
        descriptor = classify_exception(error)
        response = CommandResponse(
            command=_detect_command(argv),
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
        if _wants_json(argv):
            _emit_json(response.to_envelope())
        else:
            sys.stderr.write(f"{descriptor.message}\n")
        return descriptor.exit_code

    if args.output == "json":
        _emit_json(response.to_envelope())
    else:
        if args.command == "prompt" and response.artifacts:
            artifact = response.artifacts[0]["value"]
            if args.prompt_command == "list":
                sys.stdout.write(render_prompt_catalog_text(artifact) + "\n")
            elif args.prompt_command == "show":
                sys.stdout.write(render_prompt_template_text(artifact) + "\n")
            elif args.prompt_command == "render":
                sys.stdout.write(render_rendered_prompt_text(artifact) + "\n")
        elif args.command == "view" and response.artifacts:
            sys.stdout.write(render_view_summary(response.artifacts[0]["value"]) + "\n")
        elif (
            args.command
            in {
                "describe",
                "schema",
                "doctor",
                "evaluate",
                "analyze",
                "retrieve-cache",
            }
            and response.artifacts
        ):
            sys.stdout.write(
                json.dumps(response.artifacts[0]["value"], indent=2) + "\n"
            )
        elif response.warnings:
            sys.stdout.write("\n".join(response.warnings) + "\n")
    return response.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

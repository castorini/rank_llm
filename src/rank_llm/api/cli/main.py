from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from typing import Any, NoReturn

from rank_llm.api.adapters import make_data_artifact, serialize_data
from rank_llm.api.cli.config import load_config
from rank_llm.api.cli.prompt_view import (
    PromptTemplateError,
    build_prompt_template_view,
    build_rendered_prompt_view,
    list_prompt_templates,
    render_prompt_catalog_text,
    render_prompt_template_text,
    render_rendered_prompt_text,
)
from rank_llm.api.cli.view import ViewError, build_view_summary, render_view_summary
from rank_llm.api.error_utils import classify_exception, has_partial_success_metrics
from rank_llm.api.introspection import (
    COMMAND_DESCRIPTIONS,
    SCHEMAS,
    doctor_report,
    validate_rerank_batch_file,
    validate_rerank_payload,
)
from rank_llm.api.operations import (
    run_evaluate_aggregate,
    run_rerank,
    run_response_analysis_files,
    run_retrieve_and_rerank,
    run_retrieve_cache_generation,
)
from rank_llm.api.options import (
    RerankOptions,
    RetrievalOptions,
    add_option_arguments,
    option_values,
    validate_rerank_options,
    validate_retrieval_options,
)
from rank_llm.api.responses import CommandResponse
from rank_llm.api.spec import EXIT_CODES, KNOWN_COMMANDS, TOP_LEVEL_EXAMPLES
from rank_llm.data import RerankValidationError, normalize_rerank_input


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
    add_option_arguments(rerank_parser)
    add_option_arguments(rerank_parser, RetrievalOptions)
    rerank_parser.add_argument("--input-json", dest="input_json")
    rerank_parser.add_argument("--stdin", action="store_true")
    rerank_parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    rerank_parser.add_argument(
        "--validate-only", dest="validate_only", action="store_true"
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
    add_option_arguments(serve_http_parser)
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
    sources = [
        bool(args.dataset),
        bool(args.requests_file),
        args.input_json is not None,
        args.stdin,
    ]
    if sum(sources) == 1:
        if (args.input_json is not None or args.stdin) and args.retriever_host:
            raise CLIError(
                "direct candidates cannot be combined with retriever_host",
                exit_code=2,
                status="validation_error",
                error_code="invalid_arguments",
                command="rerank",
            )
        return
    raise CLIError(
        "Rerank requires exactly one input source: --dataset, --requests-file, --input-json, or --stdin",
        exit_code=EXIT_CODES["invalid_arguments"],
        status="validation_error",
        error_code="missing_input_source" if not any(sources) else "invalid_arguments",
        command="rerank",
    )


def _validate_rerank_execution_args(args: argparse.Namespace) -> None:
    try:
        validate_retrieval_options(
            RetrievalOptions(**option_values(args, RetrievalOptions)),
            rerank_options=RerankOptions(**option_values(args)),
            check_file=False,
        )
    except RerankValidationError as error:
        raise CLIError(
            str(error),
            exit_code=2,
            status="validation_error",
            error_code="invalid_arguments",
            command="rerank",
        ) from error


def _validate_rerank_backend_flags(args: argparse.Namespace, *, command: str) -> None:
    try:
        validate_rerank_options(RerankOptions(**option_values(args)))
    except RerankValidationError as error:
        raise CLIError(
            str(error),
            exit_code=2,
            status="validation_error",
            error_code="invalid_arguments",
            command=command,
        ) from error


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
    _validate_rerank_backend_flags(args, command="rerank")
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
        normalized = normalize_rerank_input(payload)
        results = run_rerank(options=RerankOptions(**option_values(args)), **normalized)
        input_mode = "direct"
    else:
        validation = {"valid": True, "record_count": 0, "errors": []}
        _validate_rerank_execution_args(args)
        if args.requests_file:
            validation = validate_rerank_batch_file(args.requests_file)
            if not validation["valid"]:
                return _validation_error_response("rerank", validation)
        if args.validate_only or args.dry_run:
            input_mode = (
                "service"
                if args.retriever_host
                else "requests-file"
                if args.requests_file
                else "dataset"
            )
            return CommandResponse(
                command="rerank",
                mode="validate" if args.validate_only else "dry_run",
                validation=validation,
                inputs={"mode": input_mode},
                resolved={"model_path": args.model_path, "input_mode": input_mode},
            )
        workflow = option_values(args, RetrievalOptions)
        results = run_retrieve_and_rerank(
            options=RerankOptions(**option_values(args)),
            retrieval=RetrievalOptions(**workflow),
        )
        input_mode = (
            "service"
            if args.retriever_host
            else "requests-file"
            if args.requests_file
            else "dataset"
        )

    return CommandResponse(
        command="rerank",
        validation=validation,
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
            valid = (
                isinstance(payload, dict)
                and "query" in payload
                and "candidates" in payload
            )
            validation = {
                "valid": valid,
                "record_count": 1 if valid else 0,
                "errors": []
                if valid
                else ["payload must contain query and candidates"],
            }
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
            from rank_llm.api.mcp.mcp_rankllm import run_mcp_server

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

        from rank_llm.api.rest.app import create_app
        from rank_llm.api.rest.runtime import ServerConfig
    except ModuleNotFoundError as error:
        raise CLIError(
            "serve http requires FastAPI dependencies; install the `api` extra",
            exit_code=EXIT_CODES["missing_resource"],
            status="validation_error",
            error_code="missing_api_dependencies",
            command="serve",
            details={"missing_dependencies": ["fastapi", "uvicorn"]},
        ) from error

    _validate_rerank_backend_flags(args, command="serve")

    app = create_app(
        ServerConfig(host=args.host, port=args.port, **option_values(args))
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
        # Keep diagnostics out of the machine-readable response, including
        # prints from retrieval and model initialization.
        output_context = (
            contextlib.redirect_stdout(sys.stderr)
            if args.output == "json" and args.command != "serve"
            else contextlib.nullcontext()
        )
        with output_context:
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

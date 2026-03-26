from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from typing import Any, NoReturn

from rank_llm.cli.adapters import make_data_artifact
from rank_llm.cli.config import load_config
from rank_llm.cli.introspection import (
    validate_rerank_batch_file,
    validate_rerank_payload,
)
from rank_llm.cli.operations import run_mcp_rerank, run_mcp_retrieve_and_rerank
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
    validate_parser = subparsers.add_parser("validate", help=argparse.SUPPRESS)
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

    prompt_parser = subparsers.add_parser("prompt", help=argparse.SUPPRESS)
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

    view_parser = subparsers.add_parser("view", help=argparse.SUPPRESS)
    view_parser.add_argument("path")
    view_parser.add_argument("--records", type=int, default=1)

    for command in KNOWN_COMMANDS:
        if command == "rerank":
            continue
        if command == "validate":
            continue
        if command == "prompt":
            continue
        if command == "view":
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


def _serialize_data(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            key: _serialize_data(item)
            for key, item in dataclasses.asdict(value).items()
        }
    if isinstance(value, list):
        return [_serialize_data(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_data(item) for key, item in value.items()}
    return value


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
        normalized = _normalize_direct_rerank_input(payload)
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
        artifacts=[make_data_artifact("rerank-results", _serialize_data(results))],
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


def _run_command(args: argparse.Namespace) -> CommandResponse:
    if args.command == "rerank":
        return _run_rerank_command(args)
    if args.command == "validate":
        return _run_validate_command(args)
    if args.command == "prompt":
        return _run_prompt_command(args)
    if args.command == "view":
        return _run_view_command(args)
    return CommandResponse(
        command=args.command,
        status="success",
        resolved={"config": load_config()},
        warnings=[f"{args.command} is not implemented yet."],
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        response = _run_command(args)
    except CLIError as error:
        response = _build_error_response(error)
        if _wants_json(argv):
            _emit_json(response.to_envelope())
        else:
            sys.stderr.write(f"{error.message}\n")
        return error.exit_code

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
        elif response.warnings:
            sys.stdout.write("\n".join(response.warnings) + "\n")
    return response.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

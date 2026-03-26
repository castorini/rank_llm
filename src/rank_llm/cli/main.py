from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any, NoReturn

from rank_llm.cli.config import load_config
from rank_llm.cli.responses import CommandResponse
from rank_llm.cli.spec import EXIT_CODES, KNOWN_COMMANDS, TOP_LEVEL_EXAMPLES


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
    for command in KNOWN_COMMANDS:
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


def _run_command(args: argparse.Namespace) -> CommandResponse:
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
        if response.warnings:
            sys.stdout.write("\n".join(response.warnings) + "\n")
    return response.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

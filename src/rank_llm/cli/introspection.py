from __future__ import annotations

import json
import platform
from importlib.util import find_spec
from pathlib import Path
from typing import Any

COMMAND_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "rerank": {
        "summary": "Run RankLLM reranking from direct input, dataset retrieval, or request files.",
        "input_modes": ["dataset", "requests-file", "input-json", "stdin"],
        "inspection_safe": False,
    },
    "validate": {
        "summary": "Validate rerank inputs without executing a model.",
        "targets": ["rerank"],
        "inspection_safe": True,
    },
    "prompt": {
        "summary": "Inspect bundled RankLLM prompt templates.",
        "subcommands": ["list", "show", "render"],
        "inspection_safe": True,
    },
    "view": {
        "summary": "Inspect RankLLM artifacts such as rerank JSONL, TREC runs, and invocation histories.",
        "inspection_safe": True,
    },
    "describe": {
        "summary": "Return structured metadata for a RankLLM CLI command.",
        "inspection_safe": True,
    },
    "schema": {
        "summary": "Return JSON schemas for supported inputs, outputs, and envelopes.",
        "inspection_safe": True,
    },
    "doctor": {
        "summary": "Report environment and dependency readiness for the packaged RankLLM CLI.",
        "inspection_safe": True,
    },
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "rerank-direct-input": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {"oneOf": [{"type": "string"}, {"type": "object"}]},
            "candidates": {"type": "array"},
        },
    },
    "rerank-output-record": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {"type": "object"},
            "candidates": {"type": "array"},
        },
    },
    "prompt-template": {
        "type": "object",
        "required": ["name", "template", "placeholders"],
    },
    "rendered-prompt": {
        "type": "object",
        "required": ["name", "method", "messages", "inputs"],
    },
    "view-summary": {
        "type": "object",
        "required": ["path", "artifact_type", "summary", "sampled_records"],
    },
    "doctor-output": {
        "type": "object",
        "required": [
            "python_version",
            "python_ok",
            "optional_dependencies",
            "command_readiness",
            "overall_status",
        ],
    },
    "cli-envelope": {
        "type": "object",
        "required": [
            "schema_version",
            "repo",
            "command",
            "mode",
            "status",
            "exit_code",
            "inputs",
            "resolved",
            "artifacts",
            "validation",
            "metrics",
            "warnings",
            "errors",
        ],
    },
}


def validate_rerank_payload(payload: dict[str, Any]) -> dict[str, Any]:
    valid = isinstance(payload, dict) and "query" in payload and "candidates" in payload
    return {
        "valid": valid,
        "record_count": 1 if valid else 0,
        "errors": [] if valid else ["payload must contain query and candidates"],
    }


def validate_rerank_batch_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "valid": False,
            "record_count": 0,
            "errors": [f"missing file: {path}"],
        }

    record_count = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                return {
                    "valid": False,
                    "record_count": record_count,
                    "errors": [
                        f"invalid JSON on line {line_number}: {exc.msg}",
                    ],
                }
            validation = validate_rerank_payload(payload)
            if not validation["valid"]:
                return validation
            record_count += 1
    return {"valid": True, "record_count": record_count, "errors": []}


def doctor_report() -> dict[str, Any]:
    python_version = platform.python_version()
    python_ok = tuple(int(part) for part in python_version.split(".")[:2]) >= (3, 11)
    optional_dependencies = {
        "yaml": find_spec("yaml") is not None,
        "fastmcp": find_spec("fastmcp") is not None,
        "fastapi": find_spec("fastapi") is not None,
        "pyserini": find_spec("pyserini") is not None,
    }
    command_readiness = {
        "rerank": {"ready": True},
        "prompt": {"ready": optional_dependencies["yaml"]},
        "view": {"ready": True},
        "describe": {"ready": True},
        "schema": {"ready": True},
        "doctor": {"ready": True},
    }
    overall_status = (
        "ready"
        if python_ok and all(item["ready"] for item in command_readiness.values())
        else "degraded"
    )
    return {
        "python_version": python_version,
        "python_ok": python_ok,
        "optional_dependencies": optional_dependencies,
        "command_readiness": command_readiness,
        "overall_status": overall_status,
    }

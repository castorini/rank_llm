from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommandResponse:
    command: str
    mode: str = "execute"
    status: str = "success"
    exit_code: int = 0
    inputs: dict[str, Any] = field(default_factory=dict)
    resolved: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_envelope(self) -> dict[str, Any]:
        return {
            "schema_version": "castorini.cli.v1",
            "repo": "rank_llm",
            "command": self.command,
            "mode": self.mode,
            "status": self.status,
            "exit_code": self.exit_code,
            "inputs": self.inputs,
            "resolved": self.resolved,
            "artifacts": self.artifacts,
            "validation": self.validation,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "errors": self.errors,
        }

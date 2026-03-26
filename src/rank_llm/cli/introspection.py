from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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

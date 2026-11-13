from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


class ViewError(Exception):
    pass


def build_view_summary(path: str, *, records: int = 1) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise ViewError(f"missing file: {path}")

    artifact_type = detect_artifact_type(file_path)
    if artifact_type in {"rerank-output", "request-input", "invocations-history"}:
        loaded_records = load_records(file_path, artifact_type)
        return {
            "path": str(file_path),
            "artifact_type": artifact_type,
            "summary": summarize_records(loaded_records, artifact_type),
            "sampled_records": loaded_records[:records],
        }
    if artifact_type == "trec-output":
        lines = file_path.read_text(encoding="utf-8").splitlines()
        return {
            "path": str(file_path),
            "artifact_type": artifact_type,
            "summary": {"line_count": len(lines)},
            "sampled_records": lines[:records],
        }
    raise ViewError(f"unsupported artifact type for: {path}")


def detect_artifact_type(path: Path) -> str:
    if path.suffix == ".trec":
        return "trec-output"
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and "invocations_history" in data[0]
        ):
            return "invocations-history"
    if path.suffix == ".jsonl":
        first_record = _first_jsonl_record(path)
        if "query" in first_record and "candidates" in first_record:
            candidates = first_record["candidates"]
            if any(
                not _looks_like_ranked_candidate(candidate) for candidate in candidates
            ):
                return "request-input"
            return "rerank-output"
    raise ViewError(f"could not detect artifact type for: {path}")


def load_records(path: Path, artifact_type: str) -> list[Any]:
    if artifact_type == "invocations-history":
        return cast(list[Any], json.loads(path.read_text(encoding="utf-8")))
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_view_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"path: {summary['path']}",
        f"type: {summary['artifact_type']}",
    ]
    for key, value in summary["summary"].items():
        lines.append(f"{key}: {value}")
    lines.append("[sample]")
    for record in summary["sampled_records"]:
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)


def summarize_records(records: list[Any], artifact_type: str) -> dict[str, Any]:
    if artifact_type in {"rerank-output", "request-input"}:
        candidate_total = sum(len(record.get("candidates", [])) for record in records)
        return {"record_count": len(records), "candidate_total": candidate_total}
    if artifact_type == "invocations-history":
        invocation_total = sum(
            len(record.get("invocations_history", [])) for record in records
        )
        return {"record_count": len(records), "invocation_total": invocation_total}
    return {"record_count": len(records)}


def _first_jsonl_record(path: Path) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return cast(dict[str, Any], json.loads(line))
    raise ViewError(f"empty jsonl file: {path}")


def _looks_like_ranked_candidate(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    return {"docid", "score", "doc"}.issubset(candidate)

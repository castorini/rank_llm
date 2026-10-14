from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ViewError(Exception):
    pass


def build_view_summary(path: str, *, records: int = 1) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise ViewError(f"missing file: {path}")

    artifact_type = detect_artifact_type(file_path)
    if artifact_type in {
        "rerank-output",
        "request-input",
        "invocations-history",
        "empty-json",
        "empty-jsonl",
    }:
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
        if isinstance(data, list) and not data:
            return "empty-json"
        if isinstance(data, list) and data and isinstance(data[0], dict):
            first_record = data[0]
            if "invocations_history" in first_record:
                return "invocations-history"
        else:
            raise ViewError(f"could not detect artifact type for: {path}")
    elif path.suffix == ".jsonl":
        first_record = _first_jsonl_record(path)
        if first_record is None:
            return "empty-jsonl"
    else:
        raise ViewError(f"could not detect artifact type for: {path}")
    if "query" in first_record and "candidates" in first_record:
        candidates = first_record["candidates"]
        if not isinstance(candidates, list):
            raise ViewError(f"expected a candidates list in: {path}")
        if any(not _looks_like_ranked_candidate(candidate) for candidate in candidates):
            return "request-input"
        return "rerank-output"
    raise ViewError(f"could not detect artifact type for: {path}")


def load_records(path: Path, artifact_type: str) -> list[Any]:
    if path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    else:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if artifact_type in {"rerank-output", "request-input", "invocations-history"}:
        field = (
            "invocations_history"
            if artifact_type == "invocations-history"
            else "candidates"
        )
        if not all(
            isinstance(record, dict)
            and "query" in record
            and isinstance(record.get(field), list)
            for record in records
        ):
            raise ViewError(f"expected query/{field} records in: {path}")
    return records


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


def _first_jsonl_record(path: Path) -> dict[str, Any] | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ViewError(f"expected a JSON object in: {path}")
            return record
    return None


def _looks_like_ranked_candidate(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    return {"docid", "score", "doc"}.issubset(candidate)

from __future__ import annotations

from typing import Any


def make_data_artifact(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "data",
        "value": value,
    }


def make_file_artifact(name: str, path: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "file",
        "path": path,
    }

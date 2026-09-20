from __future__ import annotations

import dataclasses
from typing import Any


def make_data_artifact(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "data",
        "value": value,
    }


def serialize_data(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [serialize_data(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_data(item) for key, item in value.items()}
    return value

from __future__ import annotations

import argparse
from collections.abc import Sequence
from enum import Enum


def translate_legacy_argv(
    argv: Sequence[str],
    *,
    mapping: dict[str, str] | None = None,
    drop_flags: set[str] | None = None,
) -> list[str]:
    translated: list[str] = []
    mapping = mapping or {}
    drop_flags = drop_flags or set()
    skip_next = False

    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if not token.startswith("--"):
            translated.append(token)
            continue

        if "=" in token:
            flag, value = token.split("=", 1)
            normalized = flag[2:]
            if normalized in drop_flags:
                continue
            mapped_flag = mapping.get(normalized, normalized.replace("_", "-"))
            translated.append(f"--{mapped_flag}={value}")
            continue

        normalized = token[2:]
        if normalized in drop_flags:
            skip_next = True
            continue
        mapped_flag = mapping.get(normalized, normalized.replace("_", "-"))
        translated.append(f"--{mapped_flag}")

    return translated


def namespace_to_legacy_argv(
    args: argparse.Namespace,
    *,
    mapping: dict[str, str] | None = None,
    drop_flags: set[str] | None = None,
) -> list[str]:
    argv: list[str] = []
    mapping = mapping or {}
    drop_flags = drop_flags or set()

    for key, value in vars(args).items():
        if key in drop_flags or value is None or value is False:
            continue
        flag = mapping.get(key, key.replace("_", "-"))
        if value is True:
            argv.append(f"--{flag}")
            continue
        if isinstance(value, list):
            if value:
                argv.append(f"--{flag}")
                argv.extend(str(item) for item in value)
            continue
        if isinstance(value, Enum):
            value = value.value
        argv.extend([f"--{flag}", str(value)])
    return argv

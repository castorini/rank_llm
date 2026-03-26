from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from rank_llm.cli.legacy import namespace_to_legacy_argv, translate_legacy_argv
from rank_llm.cli.main import main as cli_main

# Force spawn method to avoid "Cannot re-initialize CUDA in forked subprocess" error.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_DROP_FLAGS = {"prompt_mode"}


def main(args: argparse.Namespace | Sequence[str] | None = None) -> int:
    if isinstance(args, argparse.Namespace):
        _validate_backend_flags_from_namespace(args)
        argv = namespace_to_legacy_argv(args, drop_flags=_DROP_FLAGS)
    elif args is None:
        import sys

        argv = sys.argv[1:]
    else:
        argv = list(args)

    _validate_backend_flags_from_argv(argv)
    translated = translate_legacy_argv(argv, drop_flags=_DROP_FLAGS)
    return cli_main(["rerank", *translated])


def _validate_backend_flags_from_namespace(args: argparse.Namespace) -> None:
    if getattr(args, "sglang_batched", False) and getattr(
        args, "tensorrt_batched", False
    ):
        raise ValueError(
            "--sglang_batched and --tensorrt_batched are mutually exclusive"
        )


def _validate_backend_flags_from_argv(argv: Sequence[str]) -> None:
    has_sglang = any(
        token in {"--sglang_batched", "--sglang-batched"} for token in argv
    )
    has_tensorrt = any(
        token in {"--tensorrt_batched", "--tensorrt-batched"} for token in argv
    )
    if has_sglang and has_tensorrt:
        raise ValueError(
            "--sglang_batched and --tensorrt_batched are mutually exclusive"
        )


if __name__ == "__main__":
    raise SystemExit(main())

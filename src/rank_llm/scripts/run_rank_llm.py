from __future__ import annotations

import argparse
import os
import warnings
from collections.abc import Sequence

from rank_llm.cli.legacy import namespace_to_legacy_argv, translate_legacy_argv
from rank_llm.cli.main import main as cli_main

# Force spawn method to avoid "Cannot re-initialize CUDA in forked subprocess" error.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Flags accepted for backwards compatibility but not forwarded to the new CLI.
_DROP_FLAGS = {"prompt_mode"}

_PROMPT_MODE_DEPRECATION = (
    "The --prompt_mode CLI flag is deprecated and will be removed in v0.30.0. "
    "It is now ignored: prompt behavior is driven by YAML templates. "
    "Pass --prompt-template-path pointing to a template under "
    "src/rank_llm/rerank/prompt_templates/ instead "
    "(see `rank-llm prompt --help` to list the available templates)."
)


def _warn_if_prompt_mode_present(args: argparse.Namespace | Sequence[str]) -> None:
    """Emit a deprecation warning if the legacy --prompt_mode flag was supplied.

    The flag is still accepted so old invocations do not break, but it no longer
    affects behavior; YAML prompt templates replace it.
    """
    if isinstance(args, argparse.Namespace):
        present = getattr(args, "prompt_mode", None) is not None
    else:
        present = any(
            token in ("--prompt_mode", "--prompt-mode")
            or token.startswith(("--prompt_mode=", "--prompt-mode="))
            for token in args
        )
    if present:
        warnings.warn(_PROMPT_MODE_DEPRECATION, DeprecationWarning, stacklevel=3)


def main(args: argparse.Namespace | Sequence[str] | None = None) -> int:
    if isinstance(args, argparse.Namespace):
        _warn_if_prompt_mode_present(args)
        argv = namespace_to_legacy_argv(args, drop_flags=_DROP_FLAGS)
    elif args is None:
        import sys

        argv = sys.argv[1:]
        _warn_if_prompt_mode_present(argv)
    else:
        argv = list(args)
        _warn_if_prompt_mode_present(argv)

    translated = translate_legacy_argv(argv, drop_flags=_DROP_FLAGS)
    return cli_main(["rerank", *translated])


if __name__ == "__main__":
    raise SystemExit(main())

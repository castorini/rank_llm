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
        argv = namespace_to_legacy_argv(args, drop_flags=_DROP_FLAGS)
    elif args is None:
        import sys

        argv = sys.argv[1:]
    else:
        argv = list(args)

    translated = translate_legacy_argv(argv, drop_flags=_DROP_FLAGS)
    return cli_main(["rerank", *translated])



if __name__ == "__main__":
    raise SystemExit(main())

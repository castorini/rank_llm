from __future__ import annotations

import argparse
from collections.abc import Sequence

from rank_llm.cli.legacy import namespace_to_legacy_argv, translate_legacy_argv
from rank_llm.cli.main import main as cli_main


def main(args: argparse.Namespace | Sequence[str] | None = None) -> int:
    if isinstance(args, argparse.Namespace):
        argv = namespace_to_legacy_argv(args)
    elif args is None:
        import sys

        argv = sys.argv[1:]
    else:
        argv = list(args)

    translated = translate_legacy_argv(argv)
    return cli_main(["analyze", *translated])


if __name__ == "__main__":
    raise SystemExit(main())

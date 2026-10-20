"""Shared reranking-range arguments for parameterized demos."""

import argparse


def parse_args_with_rank_range(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Add a reranking range, parse arguments, and validate against retrieval depth."""
    parser.add_argument(
        "--rank-start",
        type=int,
        default=0,
        help="First reranking index (zero-based, default: 0).",
    )
    parser.add_argument(
        "--rank-end",
        type=int,
        default=None,
        help="Exclusive reranking end index (default: --k).",
    )
    args = parser.parse_args()
    if args.rank_end is None:
        args.rank_end = args.k
    if not 0 <= args.rank_start < args.rank_end <= args.k:
        parser.error("require 0 <= --rank-start < --rank-end <= --k")
    return args

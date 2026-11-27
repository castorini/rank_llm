#!/usr/bin/env python3
"""Compare two RankLLM aggregated evaluation JSONL files side by side."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: str) -> dict[str, dict[str, float]]:
    by_file: dict[str, dict[str, float]] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON on line {line_number}: {exc.msg}") from exc
        file_name = record.get("file")
        results = record.get("result", [])
        if not isinstance(file_name, str) or not isinstance(results, list):
            raise ValueError(f"{path}: malformed record on line {line_number}")
        metric_map: dict[str, float] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric")
            value = item.get("value")
            if not isinstance(metric, str):
                continue
            try:
                metric_map[metric] = float(value)
            except (TypeError, ValueError):
                continue
        by_file[file_name] = metric_map
    return by_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two RankLLM aggregated evaluation JSONL files"
    )
    parser.add_argument("--run-a", required=True, help="First aggregated JSONL file")
    parser.add_argument("--run-b", required=True, help="Second aggregated JSONL file")
    args = parser.parse_args()

    try:
        run_a = load_metrics(args.run_a)
        run_b = load_metrics(args.run_b)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    common_files = sorted(set(run_a) & set(run_b))
    if not common_files:
        print("No common evaluated files found.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Run A: {args.run_a}")
    print(f"Run B: {args.run_b}")
    print(f"Common evaluated files: {len(common_files)}")

    for file_name in common_files:
        metrics_a = run_a[file_name]
        metrics_b = run_b[file_name]
        common_metrics = sorted(set(metrics_a) & set(metrics_b))
        if not common_metrics:
            continue
        print(f"\nFile: {file_name}")
        print(f"{'Metric':<20} {'Run A':>10} {'Run B':>10} {'Delta':>10}")
        for metric in common_metrics:
            value_a = metrics_a[metric]
            value_b = metrics_b[metric]
            delta = value_b - value_a
            print(f"{metric:<20} {value_a:>10.4f} {value_b:>10.4f} {delta:>10.4f}")


if __name__ == "__main__":
    main()

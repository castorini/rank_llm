#!/usr/bin/env python3
"""Compare RankLLM evaluation JSONLs and optionally gate metric regressions.

Supports both {"metric": "ndcg_cut_10", "value": "0.75"} records and the
older trec_eval string format, "ndcg_cut_10  all  0.75".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _read_metric(item: object, path: str, line_number: int) -> tuple[str, float]:
    """Parse a structured metric or an old-style trec_eval summary line."""
    if isinstance(item, dict):
        metric, value = item.get("metric"), item.get("value")
    elif isinstance(item, str):
        parts = item.split()
        if len(parts) != 3 or parts[1] != "all":
            raise ValueError(
                f"{path}: expected 'metric all value' on line {line_number}: {item!r}"
            )
        metric, _, value = parts
    else:
        raise ValueError(f"{path}: invalid metric on line {line_number}: {item!r}")

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError(f"{path}: missing metric name on line {line_number}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: invalid value for {metric!r} on line {line_number}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{path}: non-finite value for {metric!r} on line {line_number}")
    return metric, number


def load_metrics(path: str) -> dict[str, dict[str, float]]:
    by_file: dict[str, dict[str, float]] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}: invalid JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}: expected an object on line {line_number}")
        file_name, results = record.get("file"), record.get("result")
        if not isinstance(file_name, str) or not file_name or not isinstance(results, list):
            raise ValueError(f"{path}: malformed record on line {line_number}")
        if file_name in by_file:
            raise ValueError(f"{path}: duplicate file {file_name!r} on line {line_number}")
        metric_map: dict[str, float] = {}
        for item in results:
            metric, value = _read_metric(item, path, line_number)
            if metric in metric_map:
                raise ValueError(
                    f"{path}: duplicate metric {metric!r} for {file_name!r} on line {line_number}"
                )
            metric_map[metric] = value
        by_file[file_name] = metric_map
    return by_file


def compare_metrics(
    run_a: dict[str, dict[str, float]],
    run_b: dict[str, dict[str, float]],
    *,
    selected_metrics: set[str] | None = None,
    require_complete: bool = False,
) -> list[tuple[str, str, float, float]]:
    """Return (file, metric, baseline, candidate) rows for comparable results."""
    common_files = sorted(run_a.keys() & run_b.keys())
    if not common_files:
        raise ValueError("No common evaluated files found.")
    if require_complete and run_a.keys() != run_b.keys():
        raise ValueError(
            "Run files differ; missing from B: "
            f"{sorted(run_a.keys() - run_b.keys())}; "
            f"missing from A: {sorted(run_b.keys() - run_a.keys())}"
        )

    rows: list[tuple[str, str, float, float]] = []
    for file_name in common_files:
        metrics_a, metrics_b = run_a[file_name], run_b[file_name]
        if selected_metrics is not None:
            metrics_a = {k: v for k, v in metrics_a.items() if k in selected_metrics}
            metrics_b = {k: v for k, v in metrics_b.items() if k in selected_metrics}
        if require_complete and metrics_a.keys() != metrics_b.keys():
            raise ValueError(f"Metric sets differ for {file_name!r}")
        for metric in sorted(metrics_a.keys() & metrics_b.keys()):
            rows.append((file_name, metric, metrics_a[metric], metrics_b[metric]))
    if not rows:
        raise ValueError("No common metrics found for the selected files.")
    if selected_metrics is not None and selected_metrics - {r[1] for r in rows}:
        missing = sorted(selected_metrics - {r[1] for r in rows})
        raise ValueError(f"Requested metrics not present in both runs: {missing}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare RankLLM aggregated evaluation JSONL files"
    )
    parser.add_argument("--run-a", required=True, help="Baseline evaluation JSONL")
    parser.add_argument("--run-b", required=True, help="Candidate evaluation JSONL")
    parser.add_argument(
        "--max-drop", type=float, default=None,
        help="Fail with exit code 2 if any compared metric drops by more than this amount",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help="Compare only named metrics, e.g. ndcg_cut_10 map_cut_100",
    )
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Require the same evaluated files and metric sets in both runs",
    )
    parser.add_argument(
        "--format", choices=("text", "markdown"), default="text",
        help="Report format (default: text)",
    )
    args = parser.parse_args(argv)
    if args.max_drop is not None and (
        not math.isfinite(args.max_drop) or args.max_drop < 0
    ):
        parser.error("--max-drop must be a finite, non-negative number")

    try:
        rows = compare_metrics(
            load_metrics(args.run_a),
            load_metrics(args.run_b),
            selected_metrics=set(args.metrics) if args.metrics else None,
            require_complete=args.require_complete,
        )
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    regressions = 0
    if args.format == "markdown":
        print("# RankLLM evaluation comparison")
        print(f"\nBaseline: `{args.run_a}`  ")
        print(f"Candidate: `{args.run_b}`\n")
        print("| File | Metric | Baseline | Candidate | Delta | Status |")
        print("| --- | --- | ---: | ---: | ---: | --- |")
    else:
        print(f"Run A (baseline): {args.run_a}")
        print(f"Run B (candidate): {args.run_b}")
        print(f"Compared metric rows: {len(rows)}")

    for file_name, metric, value_a, value_b in rows:
        delta = value_b - value_a
        bad = args.max_drop is not None and delta < -args.max_drop - 1e-12
        regressions += int(bad)
        status = "FAIL" if bad else "PASS" if args.max_drop is not None else "—"
        if args.format == "markdown":
            safe_file = file_name.replace("|", "\\|").replace("`", "\\`")
            print(
                f"| `{safe_file}` | {metric} | {value_a:.4f} | "
                f"{value_b:.4f} | {delta:+.4f} | {status} |"
            )
        else:
            print(
                f"{file_name}  {metric}: {value_a:.4f} -> {value_b:.4f} "
                f"(delta {delta:+.4f}) {status if args.max_drop is not None else ''}"
            )

    if args.max_drop is not None:
        summary = f"{regressions} metric regression(s) beyond allowed drop {args.max_drop:g}"
        print(f"\n{summary}")
    return 2 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())

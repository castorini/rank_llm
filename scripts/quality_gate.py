from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_BIN = Path(sys.prefix) / "bin"
CLI_SMOKE_MODULES = [
    "test.test_cli_packaging",
    "test.test_cli_scaffolding",
    "test.test_cli_rerank_command",
    "test.test_cli_validation",
    "test.test_cli_prompt",
    "test.test_cli_view",
    "test.test_cli_introspection",
    "test.test_cli_utilities",
    "test.test_cli_http",
    "test.test_cli_mcp",
    "test.test_cli_legacy_wrappers",
]


def _java_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{VENV_BIN}{os.pathsep}{env.get('PATH', '')}"
    java_executable = shutil.which("java")
    if java_executable is not None:
        env.setdefault("JAVA_HOME", str(Path(java_executable).resolve().parents[1]))
    add_modules = "--add-modules=jdk.incubator.vector"
    current_options = env.get("JAVA_TOOL_OPTIONS", "")
    if add_modules not in current_options.split():
        env["JAVA_TOOL_OPTIONS"] = f"{add_modules} {current_options}".strip()
    return env


def _run_step(
    name: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    print(f"[quality-gate] {name}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RankLLM's ordered quality gate: Ruff, tests, then mypy."
    )
    parser.add_argument(
        "--skip-ruff",
        action="store_true",
        help="Skip Ruff commands because they already ran in the current hook stage.",
    )
    args = parser.parse_args()

    python = sys.executable
    if not args.skip_ruff:
        _run_step("ruff-check", [python, "-m", "ruff", "check", "."])
        _run_step("ruff-format", [python, "-m", "ruff", "format", "--check", "."])

    _run_step(
        "analysis-tests",
        [python, "-m", "unittest", "discover", "-s", "test/analysis"],
        env=_java_env(),
    )
    _run_step(
        "evaluation-tests",
        [python, "-m", "unittest", "discover", "-s", "test/evaluation"],
        env=_java_env(),
    )
    _run_step(
        "rerank-tests",
        [python, "-m", "unittest", "discover", "-s", "test/rerank"],
        env=_java_env(),
    )
    _run_step(
        "cli-smoke",
        [python, "-m", "unittest", *CLI_SMOKE_MODULES],
        env=_java_env(),
    )
    _run_step("mypy", [python, "-m", "mypy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

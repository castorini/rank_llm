"""Run unit tests, resource-heavy integration tests, or both in order."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

INTEGRATION_MODULES = (
    "test.retrieve.test_ServiceRetriever",
    "test.test_retrieve_and_rerank",
    "test.server.test_mcp_server",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("unit", "integration", "full"))
    parser.add_argument(
        "--dry-run", action="store_true", help="show commands without running tests"
    )
    args = parser.parse_args(argv)

    phases = ("unit", "integration") if args.suite == "full" else (args.suite,)
    for phase in phases:
        env = os.environ.copy()
        # Console-entrypoint tests look up rank-llm on PATH.
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        if phase == "unit":
            env.pop("RANK_LLM_RUN_INTEGRATION_TESTS", None)
            command = [sys.executable, "-m", "unittest", "discover", "test"]
        else:
            env["RANK_LLM_RUN_INTEGRATION_TESTS"] = "1"
            command = [sys.executable, "-m", "unittest", *INTEGRATION_MODULES]

        print(f"{phase} tests: {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, env=env, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One-time environment setup for the docs/install-colab-cli.md guide.

Sent to a Colab runtime with:

    colab exec -s rankllm -f scripts/colab_setup.py --timeout 3600

Installs JDK 21 (needed by Pyserini/Anserini for first-stage retrieval and
trec_eval), clones rank_llm, and installs it with the pyserini extra.
"""

import subprocess
import sys


def sh(cmd):
    """Run a shell command, streaming its output back to the caller.

    The Colab kernel only forwards Python-level stdout, not the OS-level
    output of child processes, so we merge stderr into stdout and re-print
    line by line -- otherwise errors from apt/git/pip would be invisible.
    """
    print(f"\n>>> {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise SystemExit(f"command failed: {cmd}")


sh("apt-get update -qq && apt-get install -y -qq openjdk-21-jdk-headless")

# Start from a clean clone so re-running this file after a failure is safe.
sh(
    "rm -rf /content/rank_llm && "
    "git clone https://github.com/castorini/rank_llm.git /content/rank_llm"
)

sh(f"{sys.executable} -m pip install -q -e '/content/rank_llm[pyserini]'")

print("\nsetup ok")

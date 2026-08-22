"""One-time environment setup for the docs/install-colab-cli.md guide.

Sent to a Colab runtime with:

    colab exec -s rankllm -f scripts/colab_setup.py --timeout 3600

Installs JDK 21 (needed by Pyserini/Anserini for first-stage retrieval and
trec_eval), clones rank_llm, installs it with the pyserini extra, and
pre-seeds the DL20 qrels the eval step needs (Anserini's own qrels download
currently 404s -- see the pre-seed step below for why).
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

# Anserini's own qrels download (io.anserini.eval.RelevanceJudgments) hits a
# hardcoded anserini-tools URL that no longer resolves -- the file moved to
# a different path upstream. Pre-seed it into Pyserini's cache here so the
# eval step at the end of the pipeline finds it already there instead of
# trying (and failing) to download it. See rank_llm#422 for the same fix
# applied to the RankZephyr/FirstMistral onboarding docs.
sh(
    "mkdir -p ~/.cache/pyserini/topics-and-qrels && "
    "curl -sSL -o ~/.cache/pyserini/topics-and-qrels/qrels.dl20-passage.txt "
    "https://raw.githubusercontent.com/castorini/anserini-tools/master/qrels/qrels.dl20-passage.txt"
)

print("\nsetup ok")

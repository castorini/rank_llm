# rank_llm on Google Colab (Colab CLI recipes)

Shared plumbing for self-contained recipes that drive the
[Google Colab CLI](https://github.com/googlecolab/google-colab-cli) to run
rank_llm workloads on a remote Colab GPU, then download the results to your
machine. Nothing in the core `rank_llm` package is modified — everything lives
in this directory.

This directory currently contains the shared helpers; the runnable recipes
(`rerank_on_colab.sh`, `train_on_colab.sh`) build on top of them in follow-up
changes.

## What's here

| File | Purpose |
| --- | --- |
| `_remote_bootstrap.py` | Sent to the runtime by `colab exec -f`. Clones rank_llm, installs extras, runs the job, prints `ARTIFACT_PATH=`. |
| `_colab_common.sh` | Shared bash helpers (provision / exec / download / DRY_RUN). |

## How it works

`colab exec` only transmits **one** file and can't pass argv. So each orchestrator
builds a temp bootstrap file = an injected `CONFIG = json.loads(...)` line
prepended to `_remote_bootstrap.py`, and sends that. The runtime script then
clones rank_llm, `pip install`s the right extras, runs the command, and prints
`ARTIFACT_PATH=<remote .tar.gz>` on its last line for the caller to `colab download`.

## Prerequisites (for live use)

Install and authenticate the CLI once (Linux/macOS only):

```bash
uv tool install git+https://github.com/googlecolab/google-colab-cli
# then authenticate per the CLI's prompts
```

## Manual testing

Offline checks (no Colab account, no GPU spend):

```bash
# 1. The remote bootstrap compiles
python3 -m py_compile colab/_remote_bootstrap.py

# 2. Shell helpers parse
bash -n colab/_colab_common.sh

# 3. Optional: lint (no pip needed)
uv tool run --from shellcheck-py shellcheck -x colab/*.sh
```

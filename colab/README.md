# rank_llm on Google Colab (Colab CLI recipes)

Self-contained recipes that drive the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
to run rank_llm **fine-tuning** and **reranking** on a remote Colab GPU, then
download the results to your machine. Nothing in the core `rank_llm` package is
modified — everything lives in this directory plus one checked-in single-GPU
training config.

Full docs: [`../docs/colab.md`](../docs/colab.md).

## What's here

| File | Purpose |
| --- | --- |
| `train_on_colab.sh` | Local orchestrator: provision GPU → fine-tune (full-parameter accelerate/DeepSpeed flow) → download checkpoint `.tar.gz`. |
| `rerank_on_colab.sh` | Local orchestrator: provision GPU → `rank-llm rerank` → download `results.jsonl` + `results.trec` `.tar.gz`. |
| `_remote_bootstrap.py` | Sent to the runtime by `colab exec -f`. Clones rank_llm, installs extras, runs the job, prints `ARTIFACT_PATH=`. |
| `_colab_common.sh` | Shared bash helpers (provision / exec / download / DRY_RUN). |

## Features

- **Two recipes**: full-parameter reranker fine-tuning *and* listwise reranking inference.
- **Reranking has two modes**:
  - `dataset` (default) — end-to-end on a built-in benchmark (e.g. `dl19`) via a
    Pyserini prebuilt index; self-contained (installs JDK 21 + pyserini remotely).
  - `requests_url` — rerank-only over a candidates JSONL fetched from an http URL.
- **Everything is env-var configurable** (GPU, model, dataset, hyperparams, …) —
  see the tables in [`../docs/colab.md`](../docs/colab.md).
- **`DRY_RUN=1`** prints the exact `colab` commands instead of executing them, so
  you can verify parameters without spending GPU time.
- **Single `.tar.gz` artifacts** — output dirs are tarred remotely so a directory
  download is a single clean file.

## How it works

`colab exec` only transmits **one** file and can't pass argv. So each orchestrator
builds a temp bootstrap file = an injected `CONFIG = json.loads(...)` line
prepended to `_remote_bootstrap.py`, and sends that. The runtime script then
clones rank_llm, `pip install`s the right extras, runs the command, and prints
`ARTIFACT_PATH=<remote .tar.gz>` on its last line for the caller to `colab download`.

## Manual testing

### A. Offline checks (no Colab account, no GPU spend)

These verify syntax and the exact commands that *would* run:

```bash
# 1. The remote bootstrap compiles
python3 -m py_compile colab/_remote_bootstrap.py

# 2. Shell scripts parse
bash -n colab/train_on_colab.sh colab/rerank_on_colab.sh colab/_colab_common.sh

# 3. Optional: lint (no pip needed)
uv tool run --from shellcheck-py shellcheck -x colab/*.sh

# 4. Inspect the generated colab command sequence + injected params
DRY_RUN=1 bash colab/train_on_colab.sh
DRY_RUN=1 bash colab/rerank_on_colab.sh
DRY_RUN=1 RERANK_MODE=requests_url REQUESTS_URL=https://example.com/c.jsonl \
  bash colab/rerank_on_colab.sh
```

In the `DRY_RUN` output, confirm the order is **provision → exec → download** and
that your env-var overrides appear in the injected `CONFIG`.

You can also preview the exact file that gets sent to the runtime:

```bash
# prints the temp bootstrap path; cat it to see the injected CONFIG = json.loads(...)
DRY_RUN=1 bash -x colab/rerank_on_colab.sh 2>&1 | grep -i bootstrap
```

### B. Live checks (needs a Google account; provisions a paid GPU)

Prereq — install and authenticate the CLI once (Linux/macOS only):

```bash
uv tool install git+https://github.com/googlecolab/google-colab-cli
# then authenticate per the CLI's prompts
```

**Reranking (cheapest smoke test, ~minutes):**

```bash
bash colab/rerank_on_colab.sh        # dl19 + SPLADE++ + RankZephyr on an A100
# -> ./colab_runs/colab_rerank_out.tar.gz  (results.jsonl + results.trec)
tar xzf colab_runs/colab_rerank_out.tar.gz
rank-llm evaluate --help             # then evaluate the TREC file locally
```

**Fine-tuning (heavier; use A100-80GB or H100):**

```bash
NUM_TRAIN_EPOCHS=1 bash colab/train_on_colab.sh
# -> ./colab_runs/colab_run.tar.gz  (HF save_pretrained checkpoint)
```

> ⚠️ Single-GPU **full-parameter** 7B fine-tuning needs A100-80GB / H100.
> T4/L4 do not have enough memory (there is no LoRA path). See the GPU sizing
> table in [`../docs/colab.md`](../docs/colab.md).

Each recipe creates a named session (`rankllm-rerank` / `rankllm-train`), runs,
downloads the artifact, and **stops the runtime automatically**. Set
`KEEP_SESSION=1` to leave it running for debugging (`colab exec -s NAME`,
`colab sessions`, then `colab stop -s NAME`). `colab exec` has a 30s default
timeout, so the recipes raise it via `EXEC_TIMEOUT` (1h rerank / 4h train) —
bump it for larger jobs.

## Using with an agent

The Colab CLI ships `COLAB_SKILL.md`. With it loaded, an assistant can map a
request like *"fine-tune RankZephyr on Colab with an H100"* to
`GPU=H100 bash colab/train_on_colab.sh`.

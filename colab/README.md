# rank_llm on Google Colab (Colab CLI recipes)

Self-contained recipes that drive the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
to run rank_llm **reranking** on a remote Colab GPU, then download the results
to your machine. Nothing in the core `rank_llm` package is modified —
everything lives in this directory.

Full docs: [`../docs/colab.md`](../docs/colab.md).

## What's here

| File | Purpose |
| --- | --- |
| `rerank_on_colab.sh` | Local orchestrator: provision GPU → `rank-llm rerank` → download `results.jsonl` + `results.trec` `.tar.gz`. |
| `_remote_bootstrap.py` | Sent to the runtime by `colab exec -f`. Clones rank_llm, installs extras, runs the job, prints `ARTIFACT_PATH=`. |
| `_colab_common.sh` | Shared bash helpers (provision / exec / download / DRY_RUN). |

## Features

- **Reranking has two modes**:
  - `dataset` (default) — end-to-end on a built-in benchmark (e.g. `dl19`) via a
    Pyserini prebuilt index; self-contained (installs JDK 21 + pyserini remotely).
  - `requests_url` — rerank-only over a candidates JSONL fetched from an http URL.
- **Everything is env-var configurable** (GPU, model, dataset, …) —
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
bash -n colab/rerank_on_colab.sh colab/_colab_common.sh

# 3. Optional: lint (no pip needed)
uv tool run --from shellcheck-py shellcheck -x colab/*.sh

# 4. Inspect the generated colab command sequence + injected params
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

> GPU access depends on your Colab plan: a **free** account is usually limited to
> **T4**; `L4`/`A100`/`H100` need Colab Pro/Pro+. "Backend rejected accelerator"
> means you lack quota for that tier.

**Reranking (cheapest smoke test, ~minutes):**

```bash
# Pro/Pro+ (≥24 GB) — the default 7B listwise reranker:
GPU=A100 bash colab/rerank_on_colab.sh
# -> ./colab_runs/colab_rerank_out.tar.gz  (results.jsonl + results.trec)

# Free tier (T4) — the 7B model OOMs on 16 GB, so use a small pointwise model.
# bm25 retrieval is the most reliable (pure Lucene, no ONNX deps):
GPU=T4 MODEL_PATH=castorini/monot5-base-msmarco-10k RETRIEVAL_METHOD=bm25 \
  bash colab/rerank_on_colab.sh

# dataset mode auto-evaluates: nDCG@1/5/10 print in the terminal during the run.
tar xzf colab_runs/colab_rerank_out.tar.gz -C colab_runs   # results.jsonl + results.trec
```

(The default `SPLADE++_EnsembleDistil_ONNX` also works — the recipe auto-installs
`onnxruntime` for it — but it downloads a learned-sparse index + ONNX encoder, so
`bm25` is the quicker, more robust first demo.)

The recipe creates a named session (`rankllm-rerank`), runs, downloads the
artifact, and **stops the runtime automatically**. Set `KEEP_SESSION=1` to leave
it running for debugging (`colab exec -s NAME`, `colab sessions`, then
`colab stop -s NAME`). `colab exec` has a 30s default timeout, so the recipe
raises it via `EXEC_TIMEOUT` (default 1h) — bump it for larger jobs.

## Using with an agent

The Colab CLI ships `COLAB_SKILL.md`. With it loaded, an assistant can map a
request like *"rerank dl19 with RankZephyr on an A100"* to
`GPU=A100 bash colab/rerank_on_colab.sh`.

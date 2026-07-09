# Running rank_llm on Google Colab GPUs (Colab CLI recipes)

The [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) lets you
provision remote Colab GPU/TPU runtimes from your terminal (`colab new --gpu A100
-s NAME`), run code on them (`colab exec -f FILE -s NAME`), and pull artifacts
back (`colab download REMOTE LOCAL -s NAME`). It also ships an agent skill
(`COLAB_SKILL.md`) so assistants like Claude Code can drive it.

rank_llm's full-parameter reranker fine-tuning and 7B reranker inference both
need a GPU that many users don't have locally. These two recipes wrap the Colab
CLI so you can do both remotely without changing how rank_llm works:

- `colab/train_on_colab.sh` — full-parameter fine-tuning (the existing
  `training/train_rankllm.py` accelerate/DeepSpeed flow, on a single GPU).
- `colab/rerank_on_colab.sh` — listwise reranking with `rank-llm rerank`.

Each script provisions a runtime, clones rank_llm onto it, installs the right
extras, runs the job, tars the output, and downloads it as a single `.tar.gz`.

## Prerequisites

- **Colab CLI installed** (Linux/macOS only — Windows is not supported):
  ```bash
  uv tool install git+https://github.com/googlecolab/google-colab-cli
  ```
  Authenticate once with your Google account per the CLI's instructions.
- A clone of this repo (the scripts live in `colab/`).
- `python3` available locally (used only to build the remote config JSON).

## How it works

Each recipe runs four Colab CLI steps against one named session: `colab new
--gpu …` to provision, `colab exec -f …` to run, `colab download …` to fetch the
artifact, and `colab stop` to release the runtime (skipped with `KEEP_SESSION=1`).

`colab exec` transmits a **single** file to the runtime and provides no way to
pass arguments. So each recipe builds a temporary bootstrap file — an injected
`CONFIG = {...}` line prepended to `colab/_remote_bootstrap.py` — and sends that.
On the runtime, `_remote_bootstrap.py` clones rank_llm, `pip install`s the right
extras, runs the requested command, and prints `ARTIFACT_PATH=<remote .tar.gz>`.

`colab exec` defaults to a 30s timeout, so the recipes pass a large
`EXEC_TIMEOUT` (1h for reranking, 5h for training). Raise it for big jobs.

## Dry run (verify without spending GPU time)

Every recipe honors `DRY_RUN=1`, which prints the exact `colab` commands that
would run instead of executing them — useful for inspecting parameters before
provisioning a paid runtime:

```bash
DRY_RUN=1 bash colab/train_on_colab.sh
DRY_RUN=1 bash colab/rerank_on_colab.sh
```

## Training recipe

```bash
# Defaults: RankZephyr (zephyr-7b-beta), generation objective, 1 epoch, A100.
bash colab/train_on_colab.sh

# FirstMistral-style combined objective on an H100:
GPU=H100 \
  BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
  OBJECTIVE=combined \
  EXTRA_TRAIN_ARGS="--ranking_loss ranknet --weighted" \
  bash colab/train_on_colab.sh
```

The checkpoint (standard Hugging Face `save_pretrained` layout) is downloaded to
`./colab_runs/colab_run.tar.gz`. Unpack and run inference locally or load it from
another Colab run.

### GPU sizing reality

Single-GPU **full-parameter** fine-tuning of a 7B model is memory heavy:

| GPU            | Full-parameter 7B fine-tune     |
| -------------- | ------------------------------- |
| T4 (16 GB)     | ❌ not enough memory            |
| L4 (24 GB)     | ❌ not enough memory            |
| A100 40 GB     | ⚠️ only with ZeRO-3 CPU offload, slow |
| A100 80 GB     | ✅ recommended                  |
| H100 80 GB     | ✅ recommended                  |

The recipe uses the single-GPU config `training/configs/accel_config_single_gpu.yaml`
(DeepSpeed ZeRO-3 with CPU offload, bf16) plus `--gradient_checkpointing` and
`--low_cpu_mem_usage`, and conservative defaults
(`PER_DEVICE_TRAIN_BATCH_SIZE=1`, `GRADIENT_ACCUMULATION_STEPS=16`). Tune these
via env vars. (This repo does not provide a LoRA path, so smaller GPUs cannot be
used for the 7B models.)

The training code loads models with `attn_implementation="flash_attention_2"`,
so **flash-attn is required**, and it has no prebuilt PyPI wheel. The recipe
therefore builds it on the runtime by default (`INSTALL_FLASH_ATTN=1`), which
adds roughly **30–60 minutes** before training starts. Set
`INSTALL_FLASH_ATTN=0` only if your runtime image already ships flash-attn.

### Key training env vars

| Var | Default | Notes |
| --- | --- | --- |
| `GPU` | `A100` | `T4`/`L4`/`A100`/`H100` |
| `BASE_MODEL` | `HuggingFaceH4/zephyr-7b-beta` | HF id or path |
| `TRAIN_DATA_PATH` | `rryisthebest/rank_zephyr_training_data_alpha` | HF dataset or local JSON path on the runtime |
| `OBJECTIVE` | `generation` | `generation` / `ranking` / `combined` |
| `NUM_TRAIN_EPOCHS` | `1` | |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | |
| `GRADIENT_ACCUMULATION_STEPS` | `16` | |
| `INSTALL_FLASH_ATTN` | `1` | required by the training code; builds from source (~30–60 min). Set `0` only if the runtime already has flash-attn |
| `EXTRA_TRAIN_ARGS` | (empty) | extra `train_rankllm.py` flags |
| `REF` | `main` | branch/tag of rank_llm to clone |
| `LOCAL_OUT` | `./colab_runs` | local download dir |
| `SESSION` | `rankllm-train` | Colab session name |
| `EXEC_TIMEOUT` | `18000` | exec timeout in seconds (raise for long runs) |
| `KEEP_SESSION` | `0` | `1` keeps the runtime alive after the run |

## Reranking recipe

Two modes:

```bash
# (1) dataset mode (default): end-to-end on a built-in benchmark using a
# Pyserini prebuilt index. Self-contained — installs JDK 21 + pyserini remotely.
bash colab/rerank_on_colab.sh                       # dl19 + SPLADE++ + RankZephyr

# (2) requests_url mode: rerank-only over a candidates JSONL you host somewhere
# downloadable (HF/GCS/any http). No retrieval is run.
RERANK_MODE=requests_url \
  REQUESTS_URL=https://example.com/candidates.jsonl \
  bash colab/rerank_on_colab.sh
```

Results download to `./colab_runs/colab_rerank_out.tar.gz` (contains
`results.jsonl` and `results.trec`). In **dataset mode** rank_llm already
evaluates against the benchmark's qrels and prints **nDCG@1/5/10** during the
run, so the metric appears in the `colab exec` output — no extra step needed.

### GPU availability and sizing

Which accelerators you can provision depends on your **Colab subscription** — a
free account is typically limited to **T4** (or CPU); `L4`/`A100`/`H100` need
Colab Pro/Pro+. If `colab new` reports *"Backend rejected accelerator"*, you lack
quota for that tier; pick one your plan allows.

The default `castorini/rank_zephyr_7b_v1_full` is a **7B listwise** reranker
served with vLLM (fixed `gpu_memory_utilization=0.90`), so it needs roughly
**≥24 GB** — an L4, A100, or H100. **A 16 GB T4 cannot load it** (no room left
for the KV cache).

For a **free-tier (T4) demo**, swap in a small pointwise reranker — these run via
`transformers`, not vLLM, and fit a T4 (or even CPU):

```bash
GPU=T4 MODEL_PATH=castorini/monot5-base-msmarco-10k RETRIEVAL_METHOD=bm25 \
  bash colab/rerank_on_colab.sh
# also small: MODEL_PATH=castorini/monoelectra-base
```

`RETRIEVAL_METHOD=bm25` (pure Lucene, no ONNX) is the most reliable retriever for
a quick demo. The default `SPLADE++_EnsembleDistil_ONNX` works too — the recipe
auto-installs `onnxruntime` — but it pulls a larger learned-sparse index.

| GPU         | 7B listwise (RankZephyr) | small pointwise (monoT5-base) |
| ----------- | ------------------------ | ----------------------------- |
| T4 (16 GB)  | ❌ OOM                   | ✅                            |
| L4 (24 GB)  | ✅                       | ✅                            |
| A100/H100   | ✅ recommended           | ✅                            |

### Key reranking env vars

| Var | Default | Notes |
| --- | --- | --- |
| `GPU` | `A100` | |
| `MODEL_PATH` | `castorini/rank_zephyr_7b_v1_full` | reranker model |
| `RERANK_MODE` | `dataset` | `dataset` / `requests_url` |
| `DATASET` | `dl19` | dataset-mode benchmark |
| `RETRIEVAL_METHOD` | `SPLADE++_EnsembleDistil_ONNX` | dataset-mode retrieval |
| `REQUESTS_URL` | (empty) | required for `requests_url` mode |
| `TOP_K` | `100` | candidates per query |
| `CONTEXT_SIZE` | `4096` | |
| `NUM_GPUS` | `1` | |
| `EXTRA_RERANK_ARGS` | (empty) | extra `rank-llm rerank` flags |
| `SESSION` | `rankllm-rerank` | Colab session name |
| `EXEC_TIMEOUT` | `3600` | exec timeout in seconds (raise for long runs) |
| `KEEP_SESSION` | `0` | `1` keeps the runtime alive after the run |

## Using your own local candidates file

The `requests_url` mode fetches candidates over http from inside the runtime,
which is the simplest path. If your candidates JSONL is only on your laptop, the
CLI also has `colab upload LOCAL REMOTE -s NAME`, so you can push it onto a kept
session and point the recipe at it:

```bash
KEEP_SESSION=1 RERANK_MODE=requests_url REQUESTS_URL=file:///content/cands.jsonl \
  bash colab/rerank_on_colab.sh   # (then colab upload ... before exec, advanced)
```

For a smooth one-command demo, hosting the JSONL at a URL and using
`RERANK_MODE=requests_url` is recommended.

## Troubleshooting

- **`Backend rejected accelerator 'X'`** — your Colab plan lacks quota for that
  tier. Free accounts are typically T4-only; use `GPU=T4` (and a small pointwise
  model, see above) or upgrade to Colab Pro for `L4`/`A100`/`H100`.
- **`RuntimeError: Connection was lost.`** — the kernel websocket dropped (common
  right after a session becomes READY, or on long-held connections). The recipes
  warm up the kernel first and **retry `colab exec` automatically**
  (`EXEC_RETRIES`, default 3; `EXEC_RETRY_WAIT`, default 15s). If it still fails,
  re-run the recipe (the bootstrap is idempotent — the clone and installs are
  reused), bump `EXEC_RETRIES`, or run `colab update` to get the latest CLI.
- **`openai.OpenAIError: Missing credentials` during a pyserini run** — pyserini
  1.2.0 eagerly constructs an `openai.OpenAI()` client at import time, which
  `openai>=2` rejects when no key is set (even for bm25, which never uses it).
  The recipe sets a placeholder `OPENAI_API_KEY` for the reranking subprocess so
  the import succeeds; the client is never actually called.
- **Job outlives `EXEC_TIMEOUT`** — raise it (e.g. `EXEC_TIMEOUT=7200`).
- **Inspect a stuck run** — pass `KEEP_SESSION=1`, then attach with
  `colab exec -s <session>` / `colab sessions`, and `colab stop -s <session>`
  when done.

## Using with an agent

The Colab CLI's bundled `COLAB_SKILL.md` teaches an agent how to drive the CLI.
With that skill loaded, you can ask an assistant to run these recipes directly,
e.g. *"fine-tune RankZephyr on Colab with an H100"* maps to
`GPU=H100 bash colab/train_on_colab.sh`.

#!/usr/bin/env bash
#
# Fine-tune a rank_llm reranker on a remote Google Colab GPU runtime via the
# Colab CLI. Provisions a GPU, clones rank_llm on the runtime, installs the
# training extras, runs the existing full-parameter accelerate/DeepSpeed flow
# on a single GPU, and downloads the resulting checkpoint as a .tar.gz.
#
# Full-parameter 7B fine-tuning on a single GPU is heavy: T4/L4 are NOT enough.
# Use A100-80GB or H100. See docs/colab.md for details.
#
# Usage:
#   bash colab/train_on_colab.sh
#   GPU=H100 OBJECTIVE=combined EXTRA_TRAIN_ARGS="--ranking_loss ranknet --weighted" \
#     bash colab/train_on_colab.sh
#   DRY_RUN=1 bash colab/train_on_colab.sh   # print the colab commands only
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab/_colab_common.sh
source "$SCRIPT_DIR/_colab_common.sh"

# --- Configurable knobs (env vars) ------------------------------------------
GPU="${GPU:-A100}"
SESSION="${SESSION:-rankllm-train}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-14400}"
REPO_URL="${REPO_URL:-https://github.com/castorini/rank_llm.git}"
REF="${REF:-main}"
WORKDIR="${WORKDIR:-/content/rank_llm}"
BASE_MODEL="${BASE_MODEL:-HuggingFaceH4/zephyr-7b-beta}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-rryisthebest/rank_zephyr_training_data_alpha}"
OBJECTIVE="${OBJECTIVE:-generation}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/training/models/colab_run}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-10}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
LOCAL_OUT="${LOCAL_OUT:-./colab_runs}"

# --- Build the remote CONFIG as JSON ----------------------------------------
export RL_REPO_URL="$REPO_URL" RL_REF="$REF" RL_WORKDIR="$WORKDIR" \
  RL_BASE_MODEL="$BASE_MODEL" RL_TRAIN_DATA_PATH="$TRAIN_DATA_PATH" \
  RL_OBJECTIVE="$OBJECTIVE" RL_OUTPUT_DIR="$OUTPUT_DIR" \
  RL_NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS" \
  RL_PER_DEVICE_TRAIN_BATCH_SIZE="$PER_DEVICE_TRAIN_BATCH_SIZE" \
  RL_GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
  RL_NUM_WARMUP_STEPS="$NUM_WARMUP_STEPS" \
  RL_INSTALL_FLASH_ATTN="$INSTALL_FLASH_ATTN" \
  RL_EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS"

CONFIG_JSON="$(python3 - <<'PY'
import json, os, shlex
cfg = {
    "mode": "train",
    "repo_url": os.environ["RL_REPO_URL"],
    "ref": os.environ["RL_REF"],
    "workdir": os.environ["RL_WORKDIR"],
    "base_model": os.environ["RL_BASE_MODEL"],
    "train_dataset_path": os.environ["RL_TRAIN_DATA_PATH"],
    "objective": os.environ["RL_OBJECTIVE"],
    "output_dir": os.environ["RL_OUTPUT_DIR"],
    "num_train_epochs": int(os.environ["RL_NUM_TRAIN_EPOCHS"]),
    "per_device_train_batch_size": int(os.environ["RL_PER_DEVICE_TRAIN_BATCH_SIZE"]),
    "gradient_accumulation_steps": int(os.environ["RL_GRADIENT_ACCUMULATION_STEPS"]),
    "num_warmup_steps": int(os.environ["RL_NUM_WARMUP_STEPS"]),
    "install_flash_attn": os.environ["RL_INSTALL_FLASH_ATTN"] == "1",
    "extra_train_args": shlex.split(os.environ["RL_EXTRA_TRAIN_ARGS"]),
}
print(json.dumps(cfg))
PY
)"

REMOTE_ARTIFACT="${OUTPUT_DIR}.tar.gz"

echo "==> rank_llm Colab training recipe"
echo "    session=$SESSION GPU=$GPU model=$BASE_MODEL objective=$OBJECTIVE epochs=$NUM_TRAIN_EPOCHS"
echo "    remote output=$OUTPUT_DIR -> artifact=$REMOTE_ARTIFACT"

require_colab
TMP_BOOTSTRAP="$(build_bootstrap "$CONFIG_JSON")"
SESSION_CREATED=0
cleanup() {
  rm -f "$TMP_BOOTSTRAP"
  if [[ "$SESSION_CREATED" == "1" ]]; then
    stop_session "$SESSION"
  fi
}
trap cleanup EXIT

provision_gpu "$GPU" "$SESSION"
SESSION_CREATED=1
warmup_kernel "$SESSION"
exec_bootstrap "$TMP_BOOTSTRAP" "$SESSION" "$EXEC_TIMEOUT"
download_artifact "$REMOTE_ARTIFACT" "$LOCAL_OUT" "$SESSION"

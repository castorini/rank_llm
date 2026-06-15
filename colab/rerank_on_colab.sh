#!/usr/bin/env bash
#
# Run rank_llm listwise reranking on a remote Google Colab GPU runtime via the
# Colab CLI. Provisions a GPU, clones rank_llm, installs the vLLM inference
# extras, runs `rank-llm rerank`, and downloads the results as a .tar.gz
# (results.jsonl + results.trec).
#
# Two modes:
#   RERANK_MODE=dataset       (default) end-to-end on a built-in benchmark
#                             (e.g. dl19) using a Pyserini prebuilt index.
#                             Fully self-contained; installs JDK 21 + pyserini.
#   RERANK_MODE=requests_url  rerank-only over a candidates JSONL fetched from
#                             REQUESTS_URL (HF/GCS/any http). No retrieval.
#
# In requests_url mode the candidates JSONL is fetched over http from inside the
# runtime, so host it at a reachable URL. See docs/colab.md.
#
# Usage:
#   bash colab/rerank_on_colab.sh                       # dl19 dataset demo
#   RERANK_MODE=requests_url REQUESTS_URL=https://.../cands.jsonl \
#     bash colab/rerank_on_colab.sh
#   DRY_RUN=1 bash colab/rerank_on_colab.sh             # print colab commands
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab/_colab_common.sh
source "$SCRIPT_DIR/_colab_common.sh"

# --- Configurable knobs (env vars) ------------------------------------------
GPU="${GPU:-A100}"
SESSION="${SESSION:-rankllm-rerank}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-3600}"
REPO_URL="${REPO_URL:-https://github.com/castorini/rank_llm.git}"
REF="${REF:-main}"
WORKDIR="${WORKDIR:-/content/rank_llm}"
MODEL_PATH="${MODEL_PATH:-castorini/rank_zephyr_7b_v1_full}"
RERANK_MODE="${RERANK_MODE:-dataset}"
DATASET="${DATASET:-dl19}"
RETRIEVAL_METHOD="${RETRIEVAL_METHOD:-SPLADE++_EnsembleDistil_ONNX}"
REQUESTS_URL="${REQUESTS_URL:-}"
TOP_K="${TOP_K:-100}"
CONTEXT_SIZE="${CONTEXT_SIZE:-4096}"
NUM_GPUS="${NUM_GPUS:-1}"
EXTRA_RERANK_ARGS="${EXTRA_RERANK_ARGS:-}"
LOCAL_OUT="${LOCAL_OUT:-./colab_runs}"

if [[ "$RERANK_MODE" == "requests_url" && -z "$REQUESTS_URL" ]]; then
  echo "error: RERANK_MODE=requests_url requires REQUESTS_URL=<http url to candidates jsonl>" >&2
  exit 1
fi

# --- Build the remote CONFIG as JSON ----------------------------------------
export RL_REPO_URL="$REPO_URL" RL_REF="$REF" RL_WORKDIR="$WORKDIR" \
  RL_MODEL_PATH="$MODEL_PATH" RL_RERANK_MODE="$RERANK_MODE" \
  RL_DATASET="$DATASET" RL_RETRIEVAL_METHOD="$RETRIEVAL_METHOD" \
  RL_REQUESTS_URL="$REQUESTS_URL" RL_TOP_K="$TOP_K" \
  RL_CONTEXT_SIZE="$CONTEXT_SIZE" RL_NUM_GPUS="$NUM_GPUS" \
  RL_EXTRA_RERANK_ARGS="$EXTRA_RERANK_ARGS"

CONFIG_JSON="$(python3 - <<'PY'
import json, os, shlex
cfg = {
    "mode": "rerank",
    "repo_url": os.environ["RL_REPO_URL"],
    "ref": os.environ["RL_REF"],
    "workdir": os.environ["RL_WORKDIR"],
    "model_path": os.environ["RL_MODEL_PATH"],
    "rerank_mode": os.environ["RL_RERANK_MODE"],
    "dataset": os.environ["RL_DATASET"],
    "retrieval_method": os.environ["RL_RETRIEVAL_METHOD"],
    "requests_url": os.environ["RL_REQUESTS_URL"],
    "top_k": int(os.environ["RL_TOP_K"]),
    "context_size": int(os.environ["RL_CONTEXT_SIZE"]),
    "num_gpus": int(os.environ["RL_NUM_GPUS"]),
    "install_pyserini": True,
    "extra_rerank_args": shlex.split(os.environ["RL_EXTRA_RERANK_ARGS"]),
}
print(json.dumps(cfg))
PY
)"

REMOTE_ARTIFACT="${WORKDIR}/colab_rerank_out.tar.gz"

echo "==> rank_llm Colab reranking recipe"
echo "    session=$SESSION GPU=$GPU model=$MODEL_PATH mode=$RERANK_MODE"
if [[ "$RERANK_MODE" == "dataset" ]]; then
  echo "    dataset=$DATASET retrieval=$RETRIEVAL_METHOD top_k=$TOP_K"
else
  echo "    requests_url=$REQUESTS_URL"
fi
echo "    remote artifact=$REMOTE_ARTIFACT"

require_colab
TMP_BOOTSTRAP="$(build_bootstrap "$CONFIG_JSON")"
trap 'rm -f "$TMP_BOOTSTRAP"; stop_session "$SESSION"' EXIT

provision_gpu "$GPU" "$SESSION"
exec_bootstrap "$TMP_BOOTSTRAP" "$SESSION" "$EXEC_TIMEOUT"
download_artifact "$REMOTE_ARTIFACT" "$LOCAL_OUT" "$SESSION"

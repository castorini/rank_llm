#!/usr/bin/env bash
# Shared helpers for the rank_llm Colab CLI recipes.
# Sourced by train_on_colab.sh and rerank_on_colab.sh.
#
# Real Colab CLI surface these wrap (see `colab -h`):
#   colab new   --gpu A100 -s NAME        provision a named session
#   colab exec  -f FILE -s NAME --timeout N   run a file in that session
#   colab download REMOTE LOCAL -s NAME   pull an artifact back
#   colab stop  -s NAME                    release the runtime

COLAB_BIN="${COLAB_BIN:-colab}"
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PY="$COMMON_DIR/_remote_bootstrap.py"

# Run a `colab` subcommand, or just print it when DRY_RUN=1.
run_colab() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] %s' "$COLAB_BIN"
    printf ' %q' "$@"
    printf '\n'
  else
    "$COLAB_BIN" "$@"
  fi
}

# Fail early with an install hint if the CLI is missing (skipped in dry-run).
require_colab() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  if ! command -v "$COLAB_BIN" >/dev/null 2>&1; then
    echo "error: '$COLAB_BIN' not found." >&2
    echo "Install it with: uv tool install git+https://github.com/googlecolab/google-colab-cli" >&2
    exit 1
  fi
}

# Build a temp bootstrap file: an injected `CONFIG = json.loads(...)` line
# prepended to _remote_bootstrap.py. Echoes the temp file path.
build_bootstrap() {
  local cfg_json="$1"
  local tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/rankllm_colab_bootstrap.XXXXXX.py")"
  {
    printf 'import json\nCONFIG = json.loads(r"""%s""")\n' "$cfg_json"
    cat "$BOOTSTRAP_PY"
  } >"$tmp"
  printf '%s\n' "$tmp"
}

# Create a named session with the requested accelerator.
provision_gpu() {
  local gpu="$1" session="$2"
  echo "==> Creating Colab session '$session' (--gpu $gpu)"
  run_colab new --gpu "$gpu" -s "$session"
}

# Run `colab exec -f FILE` with retries. A freshly-READY kernel often drops the
# first websocket connection ("Connection was lost"), and long-held sockets can
# drop mid-run; the bootstrap is idempotent (clone skips if present), so retrying
# is safe. Tunable via EXEC_RETRIES / EXEC_RETRY_WAIT.
colab_exec_file() {
  local file="$1" session="$2" timeout="$3"
  local attempts="${EXEC_RETRIES:-3}" wait="${EXEC_RETRY_WAIT:-15}" n=1
  while true; do
    if run_colab exec -f "$file" -s "$session" --timeout "$timeout"; then
      return 0
    fi
    if ((n >= attempts)); then
      echo "error: 'colab exec' failed after ${attempts} attempt(s)." >&2
      return 1
    fi
    echo "   exec attempt ${n}/${attempts} failed; retrying in ${wait}s..." >&2
    sleep "$wait"
    n=$((n + 1))
  done
}

# Poke the kernel with a trivial cell so the websocket stabilizes before the real
# (long) exec. This absorbs the common post-READY "Connection was lost" race.
warmup_kernel() {
  local session="$1" tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/rankllm_warmup.XXXXXX.py")"
  printf 'print("kernel warmup ok")\n' >"$tmp"
  echo "==> Warming up kernel on '$session'"
  colab_exec_file "$tmp" "$session" 30
  local rc=$?
  rm -f "$tmp"
  return $rc
}

# The default exec timeout is 30s, far too short for installs + model runs, so
# callers pass a large value. Retries are handled by colab_exec_file.
exec_bootstrap() {
  local tmp="$1" session="$2" timeout="$3"
  echo "==> Executing remote bootstrap on '$session' (timeout ${timeout}s)"
  colab_exec_file "$tmp" "$session" "$timeout"
}

# Download a single remote artifact (a .tar.gz) into a local directory.
download_artifact() {
  local remote="$1" local_dir="$2" session="$3"
  local fname
  fname="$(basename "$remote")"
  echo "==> Downloading $remote -> $local_dir/$fname"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$local_dir"
  fi
  run_colab download "$remote" "$local_dir/$fname" -s "$session"
  echo "Done. Unpack with: tar xzf $local_dir/$fname -C $local_dir"
}

# Release the runtime (skipped when KEEP_SESSION=1 so you can debug via
# `colab exec -s NAME`). Tolerates a missing session.
stop_session() {
  local session="$1"
  if [[ "${KEEP_SESSION:-0}" == "1" ]]; then
    echo "==> KEEP_SESSION=1, leaving session '$session' running (stop with: colab stop -s $session)"
    return 0
  fi
  echo "==> Stopping Colab session '$session'"
  run_colab stop -s "$session" || true
}

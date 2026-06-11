#!/usr/bin/env bash
# Shared helpers for the rank_llm Colab CLI recipes.
# Sourced by train_on_colab.sh and rerank_on_colab.sh.

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

# Provision a runtime with the requested accelerator.
provision_gpu() {
  local gpu="$1"
  echo "==> Provisioning Colab runtime (--gpu $gpu)"
  run_colab --gpu "$gpu"
}

# Send the bootstrap to the runtime for execution.
exec_bootstrap() {
  local tmp="$1"
  echo "==> Executing remote bootstrap ($tmp)"
  run_colab exec -f "$tmp"
}

# Download a single remote artifact (a .tar.gz) into a local directory.
download_artifact() {
  local remote="$1" local_dir="$2"
  echo "==> Downloading artifact $remote -> $local_dir/"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$local_dir"
  fi
  run_colab download "$remote" "$local_dir/"
  echo "Done. Unpack with: tar xzf $local_dir/$(basename "$remote") -C $local_dir"
}

---
name: rank-llm-install
description: Set up a rank_llm development environment. Use when someone is onboarding, setting up a fresh clone, choosing extras such as cloud, api, local, or pyserini, or troubleshooting whether the packaged rank-llm CLI is ready.
---

# rank_llm Install

Development environment setup for [rank_llm](https://github.com/castorini/rank_llm) and its packaged `rank-llm` CLI.

## Prerequisites

- Python 3.11+
- Git
- Java 21 only for retrieval or evaluation workflows that use `pyserini`

## Verify Runtime

```bash
python3 --version
command -v uv
```

If `uv` is on PATH, use it silently. If not, ask once whether to install `uv` or continue with a fallback `pip` or `conda` path.

## Clone (if needed)

If no `pyproject.toml` is present:

```bash
git clone git@github.com:castorini/rank_llm.git && cd rank_llm
```

## Install (source - preferred)

### uv path

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev --extra cloud --extra api
```

### pip path

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[cloud,api]"
pip install pre-commit
```

### conda fallback

```bash
conda create -n rankllm python=3.11 -c conda-forge -y
conda activate rankllm
pip install -e ".[cloud,api]"
```

### PyPI alternative

```bash
uv pip install rank-llm
```

## Smoke Test

```bash
rank-llm --output json doctor
rank-llm --help
```

## Pre-commit (source installs)

```bash
uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
```

## Reference Files

- `references/extras.md` - Optional extras and when to add them

## Gotchas

- `rank-llm` is the canonical CLI entry point. The legacy scripts under `src/rank_llm/scripts/` remain available, but they are compatibility wrappers.
- `cloud` is the default hosted-provider stack and `api` is the lightweight HTTP server stack. This pair is the best default for most contributor workflows.
- `pyserini` requires Java 21 and is only needed for retrieval or evaluation workflows.
- `local`, `vllm`, `sglang`, and `tensorrt-llm` pull in heavier inference stacks. Do not install them by default.

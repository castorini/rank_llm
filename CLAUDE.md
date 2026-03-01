# Repo Project Instructions

## Scope
- Repository: `rank_llm`
- Primary stack: Python 3.11 package (`src/` layout, setuptools via `pyproject.toml`)
- Focus areas: retrieval + LLM reranking (listwise, pointwise, pairwise), evaluation, analysis, optional training and server integrations

## Environment and Setup
- Use Python `>=3.11` for package/runtime workflows.
- Use JDK 21 when retrieval/`pyserini` paths are involved (Anserini dependency).
- Prefer Linux/Windows for runtime; README states RankLLM is not compatible with macOS.
- Typical dev install:
```bash
pip install -e .[all]
```
- Optional extras:
  - `.[pyserini]` for retrieval
  - `.[genai]` for Gemini
  - `.[sglang]` / `.[tensorrt-llm]` for specialized backends
  - `.[training]` for finetuning workflows

## Project Structure
- Core package: `src/rank_llm/`
  - `retrieve/`: dataset/prebuilt-index and service retrieval
  - `rerank/`: listwise, pointwise, pairwise rerankers + prompt templates
  - `evaluation/`: `trec_eval`-based metrics
  - `analysis/`: response/cost analysis
  - `scripts/`: runnable wrappers (`run_rank_llm.py`, retrieval/eval helpers)
  - `server/`: Flask and MCP server entry points
- Tests: `test/` mirrored by feature area (`analysis`, `evaluation`, `rerank`, `retrieve`, `server`)
- Training assets: `training/`
- Docs: `docs/` including onboarding and release notes

## Build and Packaging
- Build backend: `setuptools.build_meta`
- Dependencies come from `requirements.txt` via dynamic config in `pyproject.toml`.
- Package data includes rerank prompt YAML templates in `src/rank_llm/rerank/prompt_templates/`.
- Release version source of truth is in `pyproject.toml` (`project.version`, `tool.bumpver.current_version`) and README release section.

## Formatting and Linting
- Install pre-commit hooks once:
```bash
pre-commit install
```
- Run full formatting/lint checks:
```bash
pre-commit run --all-files
```
- Enforced tools:
  - `black` (Python 3.11)
  - `isort` (`--profile=black`)
  - `flake8` with repo config (`--ignore=E501 --select=F401`)

## Testing
- Canonical local command (from CONTRIBUTING):
```bash
python -m unittest discover test
```
- CI (`.github/workflows/pr-format.yml`) gates on:
  1. pre-commit lint/format pass
  2. unittest suites for `test/analysis`, `test/evaluation`, `test/rerank`
- Retrieval and server tests exist locally but are not currently part of the CI matrix.
- Optional regression script for model-quality smoke checks:
```bash
bash regression_test.sh
```
Note: it runs real model pipelines and is compute/network heavy.

## Code Change Guidance
- Keep changes consistent with existing architecture under `rerank/`:
  - listwise, pointwise, and pairwise paths have separate inference handlers/classes
  - prompt behavior is template-driven (YAML in `prompt_templates/`), avoid hardcoding prompt text in code
- Prefer adding/updating tests alongside behavior changes.
- If modifying CLI behavior, validate corresponding scripts in `src/rank_llm/scripts/`.
- For retrieval-related changes, confirm compatibility with pyserini/anserini assumptions and JDK requirement.

## Contribution Workflow
- Follow `CONTRIBUTING.md` and `PULL_REQUEST_TEMPLATE.md`.
- PR expectations:
  - include docs updates when behavior changes
  - include tests for non-trivial modifications
  - run formatting + unit tests before pushing
  - run benchmarks/regression checks when efficiency or ranking quality may shift

## Training Workflow Notes
- Training uses a separate environment (`training/rank_llm_training_env.yml`) and `accelerate` launch flows.
- Reproduction scripts:
  - `training/scripts/train_rank_zephyr.sh`
  - `training/scripts/train_first_mistral.sh`
- Keep training dependency or config updates isolated to `training/` unless runtime package behavior changes.

## External Integrations and Servers
- Integrations documented in `docs/external-integrations.md` (LangChain, rerankers, LlamaIndex).
- Server-related code exists under `src/rank_llm/server/flask` and `src/rank_llm/server/mcp`; add/update tests under `test/server` when touching these modules.

## Release and Versioning
- Use bumpver config in `pyproject.toml` when performing version bumps.
- Ensure version consistency across:
  - `pyproject.toml`
  - README current/release section
  - `docs/release-notes/` entries for new releases

## Practical Checklist Before Opening a PR
1. Run `pre-commit run --all-files`.
2. Run `python -m unittest discover test`.
3. Update docs/examples if user-facing behavior changed.
4. Add/adjust tests for changed logic.
5. Confirm version/release files if change is part of a release.

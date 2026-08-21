# Contributing to RankLLM

RankLLM Contribution flow

## Pull Requests 

1. Fork + submit PRs.
2. PRs should have appropriate documentation describing change.
3. If PR makes modifications that warrant testing, provide tests
4. If change may impact efficiency, run benchmarks before change and after change for validation.
5. Every PR should be formatted. Below are the instructions to do so:
    - Bootstrap the repo-local development environment with `uv python install 3.11`, `uv venv --python 3.11`, `source .venv/bin/activate`, and `uv sync --group dev`
    - Run the following command in the project root to set up pre-commit and pre-push hooks (all commits through git UI will automatically be formatted): `uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push`
    - Install the full local validation stack before running the ordered gate: `uv pip install --python .venv/bin/python -e '.[server,cloud]'`
    - To run the full ordered gate manually, use `uv run python scripts/quality_gate.py`
    - `pre-commit` and `pre-push` now enforce the same order: Ruff, then required offline tests, then MyPy.
    - To run Ruff directly, use `uv run ruff check .` and `uv run ruff format --check .`
6. Run from the root directory the required offline tests before every push:
    - `uv run python -m unittest discover -s test/analysis`
    - `uv run python -m unittest discover -s test/evaluation`
    - `uv run python -m unittest discover -s test/rerank`
    - `uv run python -m unittest test.test_cli_packaging test.test_cli_scaffolding test.test_cli_rerank_command test.test_cli_validation test.test_cli_prompt test.test_cli_view test.test_cli_introspection test.test_cli_utilities test.test_cli_http test.test_cli_mcp test.test_cli_legacy_wrappers`
7. Run MyPy from the root directory with `uv run mypy`
8. Update the `pyproject.toml` if applicable

## Suggested PR Description

Use the following shape for PR descriptions so reviews stay consistent:

- `ref:` issue number or `N/A`
- `Summary`
- `Why`
- `Validation`
- `Follow-ups`

For packaging or CI changes, include a compact before/after note that shows the
install or workflow change clearly.

## Issues

We use GitHub issues to track public bugs and features requests. Please ensure your description is coherent and has provided all instructions to be able to reproduce the issue or to be able to implement the feature.

## License

By contributing to RankLLM, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.

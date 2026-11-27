# Contributing to RankLLM

RankLLM Contribution flow

## Pull Requests 

1. Fork + submit PRs.
2. PRs should have appropriate documentation describing change.
3. If PR makes modifications that warrant testing, provide tests
4. If change may impact efficiency, run benchmarks before change and after change for validation.
5. Every PR should be formatted. Below are the instructions to do so:
    - Bootstrap the repo-local development environment with `uv python install 3.12`, `uv venv --python 3.12`, `source .venv/bin/activate`, and `uv sync --group dev`
    - Run the following command in the project root to set up pre-commit and pre-push hooks (all commits through git UI will automatically be formatted): `uv run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push`
    - To manually make sure your code is correctly formatted and lint-clean, run `uv run pre-commit run --all-files`
    - To run Ruff directly, use `uv run ruff check .` and `uv run ruff format .`
6. Run from the root directory the unit tests with `uv run python -m unittest discover test`
7. Update the `pyproject.toml` if applicable

## Formatting Policy

Use Ruff's default formatting with a target line length of 100 characters.
Longer lines are acceptable when they improve readability; `E501` is intentionally
ignored. Line length alone is not a reason to request changes.

Use `# fmt: skip` to preserve the layout of an individual statement:

```python
result = some_function(first_argument, second_argument, third_argument)  # fmt: skip
```

For deliberately formatted blocks, use `# fmt: off` and `# fmt: on`:

```python
# fmt: off
ROWS = [
    ("short",  1,  2),
    ("longer", 3,  4),
]
# fmt: on
```

Keep exceptions narrow and use them sparingly. Both forms are respected by
`ruff format --check`.

Ruff is pinned to the same version in the development dependencies and pre-commit
hooks, which CI also runs. When updating Ruff, update both pins and `uv.lock`
together.

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

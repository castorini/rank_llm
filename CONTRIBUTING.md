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

### Integration tests

The ordinary unit test command skips tests that download prebuilt Pyserini
indexes, contact a retrieval service, or load GPU models. The mocked retrieval
parsing tests still run. To run an integration module, opt in explicitly:

```bash
RANK_LLM_RUN_INTEGRATION_TESTS=1 uv run python -m unittest test.retrieve.test_PyseriniRetriever
RANK_LLM_RUN_INTEGRATION_TESTS=1 uv run python -m unittest test.retrieve.test_ServiceRetriever
RANK_LLM_RUN_INTEGRATION_TESTS=1 uv run python -m unittest test.test_retrieve_and_rerank
RANK_LLM_RUN_INTEGRATION_TESTS=1 uv run python -m unittest test.server.test_flask_server
RANK_LLM_RUN_INTEGRATION_TESTS=1 uv run python -m unittest test.server.test_mcp_server
```

The Pyserini module needs the `pyserini` extra, JDK 21, and disk space for
prebuilt indexes. The service retriever, retrieve and rerank, and Flask modules
need a Pyserini REST service at `localhost:8081`; their reranking paths also
load model weights and need suitable GPU resources. The MCP module starts an
HTTP server and loads a Pyserini index and a Qwen model through vLLM, so it
also needs JDK 21, model downloads, and a suitable GPU. Set the opt-in variable
only for the selected module, since setting it for the full discovery command
runs every integration test.

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

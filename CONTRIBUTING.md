# Contributing to RankLLM

RankLLM Contribution flow

## Pull Requests 

1. Fork + submit PRs.
2. PRs should have appropriate documentation describing change.
3. If PR makes modifications that warrant testing, provide tests
4. If change may impact efficiency, run benchmarks before change and after change for validation.
5. Every PR should be formatted. Below are the instructions to do so:
    - Run the following command in the project root to set up pre-commit hooks (all commits through git UI will automatically be formatted): ```pre-commit install```
    - To manually make sure your code is correctly formatted, the following can be run: ```pre-commit run --all-files```

## Issues

We use GitHub issues to track public bugs and features requests. Please ensure your description is coherent and has provided all instructions to be able to reproduce the issue or to be able to implement the feature.

## License

By contributing to RankLLM, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
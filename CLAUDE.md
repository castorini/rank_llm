# Claude Development Guide for RankLLM

This document provides AI assistants (like Claude) with essential context about the RankLLM codebase design principles, coding standards, and best practices.

## Project Overview

RankLLM is a Python package for document reranking using Large Language Models. It supports:
- **Listwise rerankers**: RankGPT, RankZephyr, RankVicuna, LiT5 models
- **Pointwise rerankers**: MonoT5 suite
- **Pairwise rerankers**: DuoT5 suite
- Multiple backend integrations: OpenAI, Azure OpenAI, Google Gemini, OpenRouter, vLLM, SGLang, TensorRT-LLM

The package is designed for information retrieval research and production use, with emphasis on reproducibility, efficiency, and extensibility.

## Core Design Principles

### 1. Abstraction and Inheritance Hierarchy

The codebase uses a clear inheritance hierarchy:

```
RankLLM (abstract base)
├── ListwiseRankLLM (abstract for listwise methods)
│   ├── SafeOpenai (OpenAI/Azure integration)
│   ├── SafeGenai (Google Gemini integration)
│   ├── RankListwiseOSLLM (open-source models via vLLM/SGLang)
│   ├── RankFiDDistill (LiT5 distill models)
│   └── RankFiDScore (LiT5 score models)
├── PointwiseRankLLM (abstract for pointwise methods)
│   └── MonoT5
└── PairwiseRankLLM (abstract for pairwise methods)
    └── DuoT5
```

**When adding new models:**
- Inherit from the appropriate abstract class
- Implement all required abstract methods
- Follow the sliding window pattern for listwise rerankers
- Use the inference handler system for prompt generation

### 2. Data Model Consistency

The codebase uses dataclasses for type safety and consistency:

- `Request`: Contains a `Query` and list of `Candidate` objects (input to reranking)
- `Result`: Contains a `Query`, reranked `Candidate` list, and optional `invocations_history`
- `InferenceInvocation`: Tracks prompts, responses, and token counts for analysis

**Always:**
- Use these types for function signatures
- Preserve the structure when transforming data
- Use `dacite.from_dict()` for JSON deserialization

### 3. Prompt Template System

Prompts are defined in YAML files under `src/rank_llm/rerank/prompt_templates/`:

- Each template specifies the method type (e.g., `multiturn_listwise`, `pointwise`, `pairwise`)
- Templates use placeholders like `{query}`, `{candidate}`, `{rank}`, `{num}`
- Validation and extraction regexes are defined in templates
- The `BaseInferenceHandler` system processes templates

**When modifying prompts:**
- Create a new YAML template rather than hardcoding prompts
- Include `output_validation_regex` and `output_extraction_regex`
- Follow existing template structure (system_message, prefix, body, suffix)
- Update documentation if adding new template types

### 4. Thread Safety and Concurrency

The codebase supports parallel processing for API-based rerankers:

- Use `ThreadPoolExecutor` for I/O-bound operations (API calls)
- Protect shared state with locks (e.g., `self._key_lock` for API key cycling)
- **Critical**: When using global state (like `openai.api_key`), ensure thread safety throughout the entire operation
- Maintain result ordering when processing in parallel (use dictionaries indexed by original position)

**Best practices:**
- Create per-thread resources when possible (e.g., separate API clients)
- Lock around both state mutation AND the operations that depend on that state
- Use `as_completed()` for early result processing while maintaining order
- Always close progress bars in `finally` blocks

### 5. Sliding Window Reranking

Listwise rerankers use a sliding window algorithm:

1. Split candidates into windows of size `window_size`
2. Rerank each window independently
3. Merge windows with overlap using `stride`
4. Repeat until convergence or max passes reached

**Implementation requirements:**
- Window size must fit within context limits
- Use `permutation_pipeline` or `permutation_pipeline_batched` methods
- Track token usage for cost analysis
- Support variable passage lengths when applicable

### 6. Token Counting and Context Management

Accurate token counting is critical:

- Use model-specific tokenizers (tiktoken for OpenAI, HuggingFace tokenizers for open models)
- Implement `get_num_tokens()` for prompt token counting
- Implement `num_output_tokens()` for response estimation
- Respect `context_size` limits strictly
- Track both input and output tokens in `InferenceInvocation`

## Code Style and Standards

### Formatting

The project uses automated formatters configured in `.pre-commit-config.yaml`:

- **black**: Code formatting (line length, spacing)
- **isort**: Import sorting (profile=black for compatibility)
- **flake8**: Linting (E501 ignored, F401 enforced for unused imports)

**Before committing:**
```bash
pre-commit run --all-files
```

Set up pre-commit hooks:
```bash
pre-commit install
```

### Type Annotations

- Use type hints for all function signatures
- Prefer specific types over `Any` when possible
- Use `Optional[T]` for nullable parameters
- Use `Union[A, B]` for multiple type options (or `A | B` in Python 3.10+)
- Import types from `typing` module

Example:
```python
def rerank_batch(
    self,
    requests: List[Request],
    rank_start: int = 0,
    rank_end: int = 100,
    shuffle_candidates: bool = False,
    logging: bool = False,
    **kwargs: Any,
) -> List[Result]:
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `SafeOpenai`, `ListwiseRankLLM`)
- **Functions/Methods**: snake_case (e.g., `rerank_batch`, `run_llm`)
- **Private members**: Prefix with underscore (e.g., `self._model`, `self._context_size`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `TEMPLATES`, `ALPH_START_IDX`)

### Import Organization

Follow this order (enforced by isort):
1. Standard library imports
2. Third-party library imports
3. Local application imports

Example:
```python
import time
from typing import List, Optional

import openai
import tiktoken
from tqdm import tqdm

from rank_llm.data import Request, Result
from rank_llm.rerank.rankllm import PromptMode
```

## Testing Standards

### Test Organization

Tests mirror the source structure:
```
test/
├── rerank/
│   ├── listwise/
│   │   ├── test_SafeOpenai.py
│   │   ├── test_ListwiseInferenceHandler.py
│   │   └── test_RankListwiseOSLLM.py
│   ├── pointwise/
│   └── pairwise/
├── analysis/
├── evaluation/
└── retrieve/
```

### Test Requirements

When adding features:
- **Unit tests**: Test individual components in isolation
- **Mock external dependencies**: Use `unittest.mock.patch` for API calls
- **Test both success and failure cases**: Include tests for invalid inputs
- **Run tests before committing**: `python -m unittest discover test`

Example test pattern:
```python
import unittest
from unittest.mock import patch

class TestMyFeature(unittest.TestCase):
    def test_valid_inputs(self):
        # Test with valid data
        pass

    def test_failure_inputs(self):
        # Test with invalid data
        with self.assertRaises(ValueError):
            # code that should raise
            pass

    @patch("rank_llm.module.external_call")
    def test_with_mock(self, mock_call):
        # Test with mocked external dependency
        mock_call.return_value = "expected"
        # assertions
```

### Regression Testing

- The project includes `regression_test.sh` for end-to-end testing
- Run benchmarks before and after performance-critical changes
- Update expected results if changes are intentional

## Common Patterns

### 1. Model Initialization

Always validate inputs in `__init__`:

```python
def __init__(self, model: str, context_size: int, keys=None, **kwargs):
    if isinstance(keys, str):
        keys = [keys]  # Normalize to list
    if not keys:
        raise ValueError("Please provide API keys.")

    # Call parent constructor
    super().__init__(model=model, context_size=context_size, **kwargs)

    # Initialize instance variables
    self._keys = keys
    self._cur_key_id = 0
```

### 2. Batch Processing

Support both single and batch operations:

```python
def rerank(self, request: Request, **kwargs) -> Result:
    """Single request reranking."""
    results = self.rerank_batch(requests=[request], **kwargs)
    return results[0]

def rerank_batch(self, requests: List[Request], **kwargs) -> List[Result]:
    """Batch request reranking."""
    # Implementation handles multiple requests efficiently
    pass
```

### 3. Progress Tracking

Use tqdm for long-running operations:

```python
progress = tqdm(total=len(requests), desc="Reranking")
try:
    for request in requests:
        # process request
        progress.update(1)
finally:
    progress.close()
```

### 4. Error Handling

- Raise descriptive exceptions with context
- Use appropriate exception types (`ValueError`, `TypeError`, `FileNotFoundError`)
- Log errors with the `logging` module for debugging

```python
import logging
logger = logging.getLogger(__name__)

try:
    result = process_data(data)
except SomeError as e:
    logger.error(f"Failed to process data: {e}")
    raise ValueError(f"Invalid data format: {e}") from e
```

### 5. API Key Management

Support multiple API keys with cycling:

```python
def _get_next_key(self) -> str:
    """Cycle through available API keys."""
    key = self._keys[self._cur_key_id]
    self._cur_key_id = (self._cur_key_id + 1) % len(self._keys)
    return key
```

## File and Directory Guidelines

### Source Code Organization

```
src/rank_llm/
├── __init__.py              # Package exports
├── data.py                  # Core data models
├── rerank/
│   ├── __init__.py          # Reranker exports
│   ├── rankllm.py           # Abstract base class
│   ├── reranker.py          # High-level API
│   ├── inference_handler.py # Prompt generation
│   ├── listwise/            # Listwise implementations
│   ├── pointwise/           # Pointwise implementations
│   ├── pairwise/            # Pairwise implementations
│   └── prompt_templates/    # YAML prompt templates
├── retrieve/                # Retrieval utilities
├── evaluation/              # Evaluation metrics
├── analysis/                # Response analysis
├── api/                     # REST API server
├── scripts/                 # CLI scripts
└── demo/                    # Example usage
```

### Configuration Files

- `pyproject.toml`: Package metadata, dependencies, versioning
- `requirements.txt`: Core dependencies
- `.pre-commit-config.yaml`: Code formatting hooks
- `.gitignore`: Exclude build artifacts, model outputs, inference history

### Documentation

- `README.md`: User-facing documentation, quick start, model zoo
- `CONTRIBUTING.md`: Contribution guidelines
- `docs/`: Release notes, external integrations
- Inline docstrings: Use for all public classes and methods

## API Design Best Practices

### 1. Backward Compatibility

- Deprecate features gracefully (print warnings)
- Maintain deprecated modes for at least one major version
- Use `Optional` parameters for new features

Example:
```python
if prompt_mode:
    print("PromptMode is deprecated and will be removed in v0.30.0. "
          "Please use prompt_template_path instead.")
```

### 2. Sensible Defaults

- Provide defaults for optional parameters
- Use common values (e.g., `rank_end=100`, `window_size=20`)
- Document why defaults were chosen

### 3. Flexible Input Handling

- Accept both single items and lists where reasonable
- Normalize inputs early (e.g., convert string to list)
- Validate inputs and raise clear errors

### 4. Kwargs for Extensibility

Use `**kwargs` for optional features:

```python
def rerank_batch(self, requests: List[Request], **kwargs: Any) -> List[Result]:
    """
    Args:
        requests: The reranking requests
        **kwargs: Additional options including:
            populate_invocations_history (bool): Track inference calls
            top_k_retrieve (int): Cap for rank_end and window_size
    """
    populate_history = kwargs.get('populate_invocations_history', False)
    top_k = kwargs.get('top_k_retrieve')
```

## Performance Considerations

### 1. Batching

- Batch API calls when possible to reduce latency
- Use model-specific batching (vLLM, SGLang support native batching)
- Balance batch size with memory constraints

### 2. Caching

- Cache tokenizers (avoid repeated initialization)
- Consider caching prompts for identical queries
- Use model-specific optimizations (KV cache, flash attention)

### 3. Token Efficiency

- Use variable passage length when supported
- Truncate documents intelligently (keep most relevant content)
- Monitor token usage for cost optimization

### 4. Parallelization

- Use ThreadPoolExecutor for I/O-bound tasks (API calls)
- Use ProcessPoolExecutor for CPU-bound tasks (local model inference)
- Cap parallelism based on batch_size or available resources

## Common Pitfalls to Avoid

### 1. Thread Safety Issues

❌ **Don't:**
```python
with self._key_lock:
    openai.api_key = self._keys[self._cur_key_id]
# Lock released, another thread could change api_key here!
completion = openai.chat.completions.create(...)
```

✅ **Do:**
```python
# Create per-thread client or lock around entire operation
client = openai.OpenAI(api_key=self._keys[self._cur_key_id])
completion = client.chat.completions.create(...)
```

### 2. Inconsistent Temperature Settings

- Use `temperature=0` for reproducible reranking
- Document any non-zero temperature settings
- Don't change temperature accidentally when refactoring

### 3. Incomplete Error Handling in Parallel Code

❌ **Don't:**
```python
for future in as_completed(futures):
    results.append(future.result())  # Can raise, leaves other tasks hanging
```

✅ **Do:**
```python
for future in as_completed(futures):
    try:
        results.append(future.result())
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise
    finally:
        progress.update(1)
```

### 4. Hardcoded Prompts

❌ **Don't** hardcode prompts in Python code

✅ **Do** use YAML templates in `prompt_templates/`

### 5. Ignoring Token Limits

- Always check `get_num_tokens(prompt) <= self._context_size`
- Account for both input AND output tokens
- Handle truncation gracefully when passages exceed limits

## Release and Versioning

- Version format: MAJOR.MINOR.PATCH (e.g., v0.25.7)
- Update `pyproject.toml` version field
- Update `README.md` with current version
- Create release notes in `docs/release-notes/`
- Use `bumpver` for automated version bumping (configured in pyproject.toml)

## Integration Points

RankLLM integrates with external tools:
- **LlamaIndex**: Document reranking in RAG pipelines
- **LangChain**: Integration for conversational retrieval
- **Pyserini**: BM25 and SPLADE retrieval
- **Anserini**: Java-based retrieval (requires JDK 21)

When adding integrations:
- Maintain separation of concerns
- Make external dependencies optional
- Document integration in `docs/external-integrations.md`

## Getting Help

- Check existing code for similar patterns
- Review test files for usage examples
- See `README.md` for user-facing documentation
- Check `CONTRIBUTING.md` for contribution workflow
- File issues on GitHub for questions

## Summary Checklist for New Features

Before submitting a PR:

- [ ] Code follows black/isort/flake8 standards (run `pre-commit run --all-files`)
- [ ] Type annotations on all function signatures
- [ ] Docstrings for public classes and methods
- [ ] Unit tests added/updated
- [ ] Integration tests pass (`python -m unittest discover test`)
- [ ] Thread safety considered for concurrent code
- [ ] Token limits respected
- [ ] Backward compatibility maintained (or deprecation warnings added)
- [ ] Documentation updated (README, docstrings, release notes if needed)
- [ ] No hardcoded prompts (use YAML templates)
- [ ] Error handling includes descriptive messages
- [ ] Progress bars for long operations
- [ ] Logging for debugging (use `logging` module)

---

*This guide is maintained by the RankLLM team. Last updated: 2026-01-26*

# Optional Dependencies in rank_llm

rank_llm now supports optional dependencies to allow for lighter installations based on your specific needs.

## Installation Options

### Core Installation
Install just the core functionality without heavy ML libraries:
```bash
pip install rank_llm
```

This includes the basic ranking functionality and API integrations, but excludes:
- vLLM for local model inference
- transformers library for T5-based models

### With vLLM Support
For local model inference using vLLM:
```bash
pip install rank_llm[vllm]
```

This enables:
- `VllmHandler` for local model inference
- `RankListwiseOSLLM` and other vLLM-dependent models

### With Transformers Support
For T5-based models and transformers functionality:
```bash
pip install rank_llm[transformers]
```

This enables:
- `DuoT5` pairwise ranking model
- `MonoT5` pointwise ranking model
- `RankFiD` listwise models
- All T5-based inference handlers

### Full Installation
Install all optional dependencies:
```bash
pip install rank_llm[all]
```

This includes vLLM, transformers, and all other optional features.

## Error Handling

When you try to use functionality that requires missing optional dependencies, you'll get helpful error messages:

```python
from rank_llm.rerank.vllm_handler import VllmHandler

# If vLLM is not installed:
handler = VllmHandler(...)
# ImportError: vLLM is not installed. Please install it with: pip install rank_llm[vllm]
```

```python
from rank_llm.rerank.pairwise.duot5 import DuoT5

# If transformers is not installed:
model = DuoT5(...)
# ImportError: transformers is not installed. Please install it with: pip install rank_llm[transformers]
```

## Migration Guide

If you were previously using rank_llm and now get import errors, you likely need to install the optional dependencies:

1. **For vLLM users**: Run `pip install rank_llm[vllm]`
2. **For T5 model users**: Run `pip install rank_llm[transformers]`
3. **For both**: Run `pip install rank_llm[all]`

## Development

When developing rank_llm, install all dependencies:
```bash
pip install -e .[all]
```

This ensures you can test all functionality locally.
# RankLLM Release Notes (v0.26.0)

+ **Pyserini dependency:** v2.4.0

## Summary of Changes

- Added the `rank-llm` command-line interface for reranking,
  configuration, input validation, dry runs, prompt inspection,
  and artifact viewing.
- Added FastAPI HTTP serving and MCP serving, including
  per-request reranking overrides for the HTTP API.
- Added LiteLLM support for additional LLM providers.
- Added support for Jina Reranker v3, Qwen3 pointwise reranking,
  monoELECTRA, NVIDIA Nemotron 9B v2, and ReasonRank.
- Added asynchronous reranking, OpenAI-compatible vLLM inference,
  parallel RankGPT requests, and improved reasoning-model support.
- Added SPLADE v3 retrieval.
- Reorganized optional dependencies and deferred backend imports
  so users can install the integrations they need.
- Expanded parameterized demos and Colab onboarding guides.
- Installation and compatibility changes:
  - Requires Python 3.12 or newer.
  - Updated Pyserini to v2.4.0 and local inference dependencies
    to Transformers 5 or newer.
  - Included SentencePiece tokenizer dependencies in the `local` extra
    so monoT5 works without installing other integrations.
  - Migrated Gemini support to the Google Gen AI SDK.
  - Removed SGLang and TensorRT-LLM backend implementations.
- Bug fixes:
  - Restored FIRST logit reranking with asynchronous vLLM inference.
  - Fixed listwise context-length handling and vLLM cleanup.
  - Fixed LiteLLM backend selection across the CLI and HTTP API.
  - Bounded Gemini inference retries.
  - Fixed JSON output when appending results.
  - Corrected invocation-history fields and evaluation arguments.
  - Fixed cached retrieval parsing, prompt-template paths,
    and demo reranking ranges.
  - Improved validation of direct reranking inputs.

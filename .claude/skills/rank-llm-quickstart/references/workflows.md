# rank_llm Workflow Guide

## Default Hosted Workflow

Use this when the user wants the common contributor setup:

- install `cloud` and `api`
- run hosted rerankers such as OpenAI-compatible or Gemini models
- use `serve http` if needed

## Retrieval or Evaluation Workflow

Add `pyserini` when the task includes:

- dataset-backed retrieval paths that depend on Pyserini
- `trec_eval`-based evaluation over standard collections
- reconstruction of cached retrieval results from existing TREC runs

This path also requires Java 21.

## Local Model Workflow

Add `local` when the user wants:

- MonoT5, DuoT5, MonoELECTRA, or other local Hugging Face paths
- local PyTorch or transformers inference without hosted APIs

Then add a batched backend only if requested:

- `vllm` for vLLM
- `sglang` for SGLang
- `tensorrt-llm` for TensorRT-LLM

## API and MCP Workflow

- `api` is the lighter HTTP-serving stack and is enough for `rank-llm serve http`
- `mcp` is for `rank-llm serve mcp`
- `server` aggregates `api` and `mcp`

## Training Workflow

Add `training` only when touching finetuning or reproduction scripts under `training/`.

---
name: rank-llm-quickstart
description: Use when working with the rank-llm CLI: rerank, evaluate, analyze, retrieve-cache, serve, validate, prompt, view, describe, schema, or doctor. Covers entry points, common flags, JSONL and TREC artifacts, and end-to-end retrieval plus reranking workflows.
---

# rank_llm Quickstart

Reference for the packaged `rank-llm` CLI.

## CLI Entry Point

```bash
rank-llm <command> [options]
```

## Primary Commands

| Command | Purpose |
| --- | --- |
| `rerank` | Run reranking from dataset retrieval, request files, or direct JSON input |
| `evaluate` | Aggregate `trec_eval` metrics across stored rerank outputs |
| `analyze` | Analyze stored responses and error counts |
| `retrieve-cache` | Build cached retrieval JSON from an existing TREC run |
| `serve http` | Start the HTTP server |
| `serve mcp` | Start the MCP server |

## Introspection Commands

| Command | Purpose |
| --- | --- |
| `doctor` | Check Python version and optional dependency readiness |
| `describe <cmd>` | Return structured command metadata |
| `schema <name>` | Print JSON Schema for supported inputs and outputs |
| `validate rerank` | Validate rerank inputs without executing a model |
| `prompt list\|show\|render` | Inspect bundled prompt templates |
| `view <path>` | Inspect rerank JSONL, request JSONL, TREC runs, and invocation histories |

## Quick Workflow

```bash
# 1. Check environment
rank-llm doctor

# 2. Run retrieval + reranking
rank-llm rerank --model-path castorini/rank_zephyr_7b_v1_full --dataset dl20 \
  --retrieval-method bm25 --top-k-candidates 100 \
  --output-jsonl-file rerank_results.jsonl --output-trec-file rerank_results.trec

# 3. Inspect the artifact
rank-llm view rerank_results.jsonl

# 4. Aggregate trec_eval metrics from stored outputs
rank-llm evaluate --model-name castorini/rank_zephyr_7b_v1_full

# 5. Analyze invocation histories or stored response files
rank-llm analyze --files demo_outputs/inference_invocations_history.json --verbose
```

## Reference Files

Read these on demand for details:

- `references/cli-examples.md` - Common invocations for each command
- `references/input-output-examples.md` - JSONL, TREC, and invocation-history artifact shapes
- `references/workflows.md` - Backend and workflow selection guide

## Key Concepts

- **Input modes**: `rerank` accepts dataset retrieval, request files, direct JSON payloads, or stdin.
- **Artifact families**: the CLI works with request JSONL, rerank JSONL, TREC runs, invocation histories, and aggregated evaluation JSONL.
- **Hosted vs local backends**: hosted provider paths usually need `cloud`; local model paths usually need `local` or a batched backend extra.
- **Prompt templates**: rerank behavior is template-driven. Inspect bundled templates with `rank-llm prompt`.

## Gotchas

- `rerank` requires one input source: `--dataset`, `--requests-file`, `--input-json`, or `--stdin`.
- Dataset-backed `rerank` also requires `--retrieval-method`.
- `rank-llm view` detects `.trec`, `.jsonl`, and invocation-history `.json` artifacts, but it does not inspect arbitrary JSON files.
- `evaluate` operates on stored rerank outputs in a directory and writes `trec_eval_aggregated_results_<model>.jsonl`.
- `analyze` can return `partial_success` when a file mixes valid and malformed model outputs.
- `serve http` needs the `api` extra. `serve mcp` needs the `mcp` stack.

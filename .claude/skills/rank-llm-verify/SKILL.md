---
name: rank-llm-verify
description: Use when validating rank_llm artifacts after rerank, retrieve-cache, or related CLI workflows. Checks JSONL integrity, TREC formatting, candidate shape, invocation-history structure, and duplicate query identifiers. Wraps rank-llm view plus custom assertions.
---

# rank_llm Verify

Validates stored RankLLM artifacts for correctness and structural consistency.

## When to Use

- After `rank-llm rerank`
- After `rank-llm retrieve-cache`
- Before using rerank outputs for `evaluate` or `analyze`
- When comparing artifacts across models or prompt templates

## What It Checks

### JSONL Integrity

- Every line is valid JSON
- No empty files
- No truncated or malformed records

### Request Input

- Every record has `query` and `candidates`
- Every record has at least one candidate
- Candidate entries expose either `doc` or `text`

### Rerank Output

- Every record has `query` and `candidates`
- Every candidate has `docid`, `score`, and `doc`
- No duplicate query identifiers when `qid` values are present

### Invocation History

- Top-level file is a JSON list
- Each record has an `invocations_history` list
- Every invocation entry is object-shaped

### TREC Output

- Every non-empty line has 6 columns
- Rank field is an integer
- Score field is numeric

## Usage

Run the verification script:

```bash
bash .claude/skills/rank-llm-verify/scripts/verify.sh <artifact-path> [artifact-type]
```

Supported artifact types:
- `request-input`
- `rerank-output`
- `invocations-history`
- `trec-output`

If no artifact type is provided, the script attempts to auto-detect it with `rank-llm view`.

## Verification Script

See `scripts/verify.sh` for the runnable verification wrapper.

## Gotchas

- `rank-llm validate rerank` checks input contracts before execution. The verify script checks stored output artifacts after execution.
- `rank-llm view` only detects supported artifact families. If a file is not one of those shapes, pass the artifact type manually or inspect the file directly.
- A rerank JSONL file can be structurally valid and still be low quality. Use `evaluate` and `analyze` for score and response diagnostics.

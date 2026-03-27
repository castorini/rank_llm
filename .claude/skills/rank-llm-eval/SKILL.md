---
name: rank-llm-eval
description: Use when analyzing rank_llm evaluation outputs across runs or models. Covers aggregated trec_eval JSONL files, response-analysis metrics, retrieval-cache handoff files, and side-by-side comparison of stored evaluation artifacts.
---

# rank_llm Eval

Analyze and compare RankLLM evaluation outputs across models, prompt templates, or retrieval settings.

## When to Use

- After `rank-llm evaluate`
- After `rank-llm analyze`
- When comparing two rerank runs or model variants
- When checking whether a retrieval-cache or rerank artifact is ready for downstream evaluation

## What It Does

### Aggregated Metric Comparison

- Load two `trec_eval_aggregated_results_<model>.jsonl` files
- Compare per-file metric values side by side
- Report deltas for metrics such as `ndcg_cut_10`, `map_cut_100`, or recall-style scores

### Response Analysis Interpretation

- Read `analyze` summaries and error counts
- Flag mixed valid and malformed outputs when `partial_success` appears
- Separate response-format problems from ranking-quality problems

### Retrieval Handoff Checks

- Confirm that a run has the expected rerank JSONL or TREC artifacts before aggregation
- Confirm that cached retrieval JSON exists when evaluation depends on precomputed retrieval results

## Usage

Compare two evaluation outputs:

```bash
python3 .claude/skills/rank-llm-eval/scripts/compare.py \
  --run-a trec_eval_aggregated_results_a.jsonl \
  --run-b trec_eval_aggregated_results_b.jsonl
```

Or use the CLI directly:

```bash
rank-llm evaluate --model-name castorini/rank_zephyr_7b_v1_full
rank-llm analyze --files invocations.json --verbose
rank-llm view rerank_results.jsonl
```

## Reference Files

- `references/metrics.md` - Metric names, output files, and interpretation guidance

## Comparison Script

See `scripts/compare.py` for the side-by-side comparison tool.

## Gotchas

- `evaluate` aggregates stored rerank outputs. It does not execute reranking by itself.
- Comparing two aggregated evaluation files only makes sense when both runs cover the same datasets or TREC output files.
- `analyze` focuses on response and parsing quality, not ranking quality. A clean `analyze` result does not imply good `ndcg_cut_10`.
- `partial_success` means some responses were usable and some were malformed. Treat that as a data-quality issue before reading too much into the metric deltas.

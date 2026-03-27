# rank_llm Evaluation Outputs

## `rank-llm evaluate`

The CLI aggregates `trec_eval` metrics from stored rerank outputs and writes:

```text
trec_eval_aggregated_results_<model>.jsonl
```

Each JSONL record contains:

- `file`: the evaluated run file
- `result`: a list of metric objects

Typical metric names include:

- `ndcg_cut_10`
- `map_cut_100`
- `recall_20`

Interpretation:

- prefer comparing the same metric across the same evaluated files
- do not compare a DL19 run to a DL20 run and treat the delta as meaningful

## `rank-llm analyze`

The CLI reports response-analysis metrics over stored files. These are useful for:

- malformed output counts
- parsing failures
- coarse sanity checks on invocation quality

Interpretation:

- good analysis metrics do not guarantee strong ranking metrics
- `partial_success` means at least some records were malformed but the command still extracted usable information

## Evaluation Readiness Checklist

- rerank output exists and is structurally valid
- TREC run file exists when downstream tooling expects one
- aggregated evaluation files being compared cover the same benchmark slices
- response analysis does not reveal systemic parsing failures

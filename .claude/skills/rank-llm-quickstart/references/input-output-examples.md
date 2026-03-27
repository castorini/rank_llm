# rank_llm Artifact Examples

## Request JSONL

One JSON object per line:

```json
{"query":{"text":"what is bm25","qid":"q1"},"candidates":[{"docid":"D1","score":12.4,"doc":"BM25 is a sparse retrieval scoring function."},{"docid":"D2","score":11.1,"doc":"Dense retrieval uses vector similarity."}]}
```

## Rerank Output JSONL

One JSON object per line:

```json
{"query":{"text":"what is bm25","qid":"q1"},"candidates":[{"docid":"D1","score":1.0,"doc":"BM25 is a sparse retrieval scoring function."},{"docid":"D2","score":0.5,"doc":"Dense retrieval uses vector similarity."}]}
```

`rank-llm view` treats a JSONL file as rerank output when each candidate already looks like a ranked record with `docid`, `score`, and `doc`.

## Invocation History JSON

```json
[
  {
    "query": {"text": "what is bm25", "qid": "q1"},
    "invocations_history": [
      {
        "prompt": "...",
        "response": "...",
        "input_token_count": 120,
        "output_token_count": 16
      }
    ]
  }
]
```

## TREC Run Output

```text
q1 Q0 D1 1 12.4 rank_llm
q1 Q0 D2 2 11.1 rank_llm
```

## Aggregated Evaluation JSONL

`rank-llm evaluate` writes `trec_eval_aggregated_results_<model>.jsonl` records shaped like:

```json
{"file":"rerank_results.txt","result":[{"metric":"ndcg_cut_10","value":"0.7031"},{"metric":"map_cut_100","value":"0.4120"}]}
```

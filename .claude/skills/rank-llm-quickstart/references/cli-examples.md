# rank_llm CLI Examples

## Rerank

### Dataset-backed retrieval plus reranking

```bash
rank-llm rerank \
  --model-path castorini/rank_zephyr_7b_v1_full \
  --dataset dl20 \
  --retrieval-method SPLADE++_EnsembleDistil_ONNX \
  --top-k-candidates 100 \
  --prompt-template-path src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml \
  --context-size 4096 \
  --variable-passages
```

### Request-file batch reranking

```bash
rank-llm rerank \
  --model-path gpt-4o \
  --requests-file requests.jsonl \
  --top-k-candidates 100 \
  --output-jsonl-file rerank_results.jsonl \
  --output-trec-file rerank_results.trec \
  --invocations-history-file invocations.json
```

### Direct JSON reranking

```bash
rank-llm rerank \
  --model-path gpt-4o-mini \
  --input-json '{"query":"who wrote neuromancer","candidates":["William Gibson wrote Neuromancer.","Neal Stephenson wrote Snow Crash."]}'
```

### Validate without executing a model

```bash
rank-llm validate rerank --requests-file requests.jsonl
rank-llm rerank --model-path gpt-4o --requests-file requests.jsonl --validate-only
```

## Prompt Inspection

```bash
rank-llm prompt list
rank-llm prompt show rank_zephyr
rank-llm prompt render rank_zephyr --input-json '{"query":"what is bm25","candidates":["BM25 is a sparse retrieval scoring function."]}'
```

## Artifact Inspection

```bash
rank-llm view rerank_results.jsonl
rank-llm view rerank_results.trec
rank-llm view invocations.json --records 2
```

## Evaluation and Analysis

```bash
rank-llm evaluate --model-name castorini/rank_zephyr_7b_v1_full
rank-llm analyze --files invocations.json --verbose
```

## Retrieve Cache

```bash
rank-llm retrieve-cache \
  --trec-file run.trec \
  --collection-file collection.tsv \
  --query-file queries.tsv \
  --output-file retrieval_cache.json \
  --output-trec-file retrieval_cache.trec \
  --topk 20
```

## Servers

```bash
rank-llm serve http --model-path castorini/rank_zephyr_7b_v1_full --port 8082
rank-llm serve mcp --transport stdio
```

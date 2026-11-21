# KaLM-Reranker

KaLM encodes each passage independently of the query, mean-pools groups of four
encoder tokens, and uses decoder cross-attention to score relevance. RankLLM uses
the checkpoint's published Transformers scorer for those steps and its shared
pointwise pipeline for batching and ranking.
[Model card](https://huggingface.co/KaLM-Embedding/KaLM-Reranker-V1-Nano-R2)

Use RankLLM's local extra and pass the Hugging Face model ID or a local checkpoint
directory to the existing CLI. The checkpoint directory includes the model's
Python scorer, which RankLLM loads at runtime. The bundled prompt YAML provides
the task instruction. The reference scorer uses up to 512 query tokens and up
to 1,024 passage tokens. CLI options such as `--context-size` and
`--max-passage-words` can lower the passage limit.

Use `KaLMReranker(model, encoder_chunk_size=8)` in Python to change the
compression ratio, or pass `None` to disable pooling. The CLI uses `r=4`.
The current `--pointwise-vllm` backend uses chat completions with token
logprobs. The KaLM vLLM plugin exposes its own scoring endpoints.

## Reproduce TREC DL19 passage reranking

Run both models on Castorini's fixed SPLADE++ EnsembleDistil top-100 requests.
The commands use the Hugging Face CLI, RankLLM's local extra, and a standalone
`trec_eval` executable. Set `KALM_MODEL` to a local copy of the pinned
checkpoint to reuse it.

```bash
set -e
OUT=dl19-splade-top100
mkdir -p "$OUT"

CANDIDATES=$(hf download castorini/rank_llm_data \
  retrieve_results/SPLADE_P_P_ENSEMBLE_DISTIL/retrieve_results_dl19_top100.jsonl \
  --repo-type dataset --revision 39db0a25a552dd2a12012df22029686bf7fdc050)
KALM_MODEL="${KALM_MODEL:-$(hf download KaLM-Embedding/KaLM-Reranker-V1-Nano-R2 \
  --revision 7b9c7a522cdbf5b52674fa5027e1b5be64cb989f)}"
MONOT5_MODEL=$(hf download castorini/monot5-base-msmarco-10k \
  --revision f15657ab3d2a5dd0b9a30c8c0b6a0a73c9cb5884 --exclude flax_model.msgpack)
curl -fsSL https://trec.nist.gov/data/deep/2019qrels-pass.txt -o "$OUT/qrels.txt"

rank-llm rerank --requests-file "$CANDIDATES" --model-path "$KALM_MODEL" \
  --top-k-candidates 100 --top-k-rerank 100 --batch-size 32 --context-size 1024 \
  --output-trec-file "$OUT/kalm-nano-r2.trec"
rank-llm rerank --requests-file "$CANDIDATES" --model-path "$MONOT5_MODEL" \
  --top-k-candidates 100 --top-k-rerank 100 --batch-size 32 --context-size 512 \
  --output-trec-file "$OUT/monot5-base.trec"

for model in kalm-nano-r2 monot5-base; do
  trec_eval -c -q -m ndcg_cut.10 "$OUT/qrels.txt" "$OUT/$model.trec" \
    > "$OUT/$model.ndcg10.tsv"
done
```

Each metrics file contains per-query nDCG@10 and the aggregate on its `all` row.

### Reference results

An NVIDIA H100 run with PyTorch 2.10.0 and Transformers 5.6.2 produced these
scores over all 43 judged queries. Each model reranked the same 4,300 candidates.

| Model | nDCG@10 |
| --- | ---: |
| KaLM Nano-R2 (r=4) | 0.7627 |
| MonoT5-base (MS MARCO 10k) | 0.7065 |

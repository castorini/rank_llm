# GPU Variability

## Reranking Effectiveness with RankZephyr (2026/11/27)

These experiments document the score variability observed on different GPUs and configurations that yield differences in effectiveness.
In all cases, we're using RankZephyr, evaluating on TREC DL19 and DL20, something like this:

```bash
rank-llm rerank \
  --model-path castorini/rank_zephyr_7b_v1_full \
  --dataset dl19 \
  --retrieval-method SPLADE++_EnsembleDistil_ONNX \
  --top-k-candidates 100 \
  --context-size 4096
```

We use `--batch-size` to control the batch size, trying 32 and 1.

Yes, the temperature is zero in all cases.

### NVIDIA GeForce RTX 4090

First up is an NVIDIA GeForce RTX 4090 with CUDA v13.2, running `vllm` 0.20.1.

These are the results, over ten trials:

| Dataset | Batch size | `ndcg_cut_10` | Runtime |
| ------- | ---------: | ------------: | ------: |
| DL19    | 1          | 0.7775        | 11:29   |
| DL19    | 32         | 0.7805        | 01:54   |
| DL20    | 1          | 0.8178        | 14:23   |
| DL20    | 32         | 0.8178        | 02:22   |

The runtime is the time spent in running the sliding windows.

No surprise, larger batch sizes increase performance (i.e., decrease search time).
However, batch size also affects effectiveness.

Within each batch size setting, the run files are (as LLMs like to say) byte-identical.

### NVIDIA RTX PRO 6000 Blackwell

Next up, an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
Running CUDA v13.2, same version of `vllm`:

These are the results, over ten trials:

| Dataset | Batch size | Mean `ndcg_cut_10` | Sample SD | Min–max       | Runtime |
| ------- | ---------: | -----------------: | --------: | ------------: | ------: |
| DL19    | 1          | 0.7834             | 0         | 0.7834–0.7834 | 08:05   |
| DL19    | 32         | 0.7793             | 0.00315   | 0.7752–0.7847 | 01:19   |
| DL20    | 1          | 0.8213             | 0         | 0.8213–0.8213 | 10:09   |
| DL20    | 32         | 0.8133             | 0.00131   | 0.8112–0.8162 | 01:38   |

Here we see some non-determinism with a batch size of 32, as well as lower effectiveness.

The raw data to provide a sense of the observed score variability:

```bash
# DL19, batch size=32
zephyr-runs-batch32.dl19/run-003.log:ndcg_cut_10            all     0.7847
zephyr-runs-batch32.dl19/run-010.log:ndcg_cut_10            all     0.7822
zephyr-runs-batch32.dl19/run-006.log:ndcg_cut_10            all     0.7815
zephyr-runs-batch32.dl19/run-008.log:ndcg_cut_10            all     0.7809
zephyr-runs-batch32.dl19/run-001.log:ndcg_cut_10            all     0.7804
zephyr-runs-batch32.dl19/run-005.log:ndcg_cut_10            all     0.7783
zephyr-runs-batch32.dl19/run-002.log:ndcg_cut_10            all     0.7779
zephyr-runs-batch32.dl19/run-004.log:ndcg_cut_10            all     0.7767
zephyr-runs-batch32.dl19/run-007.log:ndcg_cut_10            all     0.7752
zephyr-runs-batch32.dl19/run-009.log:ndcg_cut_10            all     0.7752

# DL20, batch size=32
zephyr-runs-batch32.dl20/run-002.log:ndcg_cut_10            all     0.8162
zephyr-runs-batch32.dl20/run-010.log:ndcg_cut_10            all     0.8141
zephyr-runs-batch32.dl20/run-003.log:ndcg_cut_10            all     0.8140
zephyr-runs-batch32.dl20/run-009.log:ndcg_cut_10            all     0.8132
zephyr-runs-batch32.dl20/run-004.log:ndcg_cut_10            all     0.8131
zephyr-runs-batch32.dl20/run-005.log:ndcg_cut_10            all     0.8131
zephyr-runs-batch32.dl20/run-006.log:ndcg_cut_10            all     0.8128
zephyr-runs-batch32.dl20/run-007.log:ndcg_cut_10            all     0.8126
zephyr-runs-batch32.dl20/run-001.log:ndcg_cut_10            all     0.8125
zephyr-runs-batch32.dl20/run-008.log:ndcg_cut_10            all     0.8112
```

Why are we seeing this?

The differences between the GeForce RTX 4090 and RTX PRO 6000 Blackwell Max-Q are not surprising, since they represent different architectures (Ada Lovelace vs. Blackwell).
Across different GPUs, architecture-dependent numerical differences represent a plausible explanation for the different effectiveness scores, even at batch size 1.

As for the batch size behavior in the RTX PRO 6000 Blackwell?
As explained by Codex:
One possible culprit is concurrent inference in vLLM.
In `listwise_rankllm.py`, the ranking scheduler processes up to `--batch-size` windows concurrently.
As each window completes, it submits the next window for that query.
The `vllm_handler.py` uses continuous batching.

Different completion timing can therefore change which prompts share a GPU batch.
Batch-dependent floating-point differences can change the highest-scoring token when logits are close, despite greedy decoding.
vLLM explicitly says its default configuration does not guarantee reproducibility ([vLLM documentation](https://docs.vllm.ai/en/stable/usage/reproducibility/)).

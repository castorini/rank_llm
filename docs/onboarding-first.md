# RankLLM: Listwise Reranking with FirstMistral

This guide presents a gentle introduction to reranking, the role it plays in the multi-stage retrieval pipeline, and how LLMs are emerging as a powerful tool for reranking.

If you're a Waterloo student traversing the [onboarding path](https://github.com/lintool/guide/blob/master/ura.md),
make sure you've first done all the exercises leading up to this guide, starting [here](https://github.com/castorini/anserini/blob/master/docs/start-here.md).
The [previous step](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md) in the onboarding path is to implement sparse and dense retrieval in Pyserini.
In general, don't try to rush through this guide by just blindly copying and pasting commands into a shell;
that's what I call [cargo culting](https://en.wikipedia.org/wiki/Cargo_cult_programming).
Instead, really try to understand what's going on.

The experiments in this guide require a GPU.
Before starting, work through [install-colab-cli.md](install-colab-cli.md), which sets up the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) and has you run a BM25 + monoT5 reranking pipeline on a free Colab GPU, driven entirely from your terminal.

**Learning outcomes** for this guide:

- Understand the motivation and architecture of multi-stage retrieval.
- Understand listwise reranking with LLMs, using sliding windows.
- Understand how FIRST speeds up the reranking process by leveraging logits.
- Be able to run end-to-end multi-stage retrieval pipelines with RankZephyr and FirstMistral.

In this guide, we will focus on the listwise approach.
For more information about pointwise and pairwise, one can refer to [Zhuang et al. (2024)](https://arxiv.org/abs/2310.14122) and [Qin et al. (2024)](https://arxiv.org/abs/2306.17563).

> Note: as you can tell from the years of the citations, reranking with LLMs is quite a recent topic; indeed, this is still a highly active area of research. Thus, beware that the "knowledge-cutoff" of this guide is Jan 2025.

## Understanding FIRST: First-Token Reranking

FIRST (Faster Improved Listwise Reranking with Single Token Decoding) is a novel approach to reranking with LLMs that is up to 42% faster to inference than the "traditional" approach we have presented above for RankZephyr.

At a high level, instead of prompting the LLM to generate a full ranking of the documents (e.g. "[3] > [1] > [2]"), we examine the probability that each document will be ranked as the top document by the LLM, and infer the ranking from these probabilities.
For example, if the probabilities of ranking documents 1, 2, 3 as the top document are 0.2, 0.1, 0.7, respectively, then we hypothesize that the true ranking is [3] > [1] > [2] without having to wait for the LLM to generate the full ranking in text, avoiding a major bottleneck in inference efficiency.
> How do we obtain such probabilities? We use the logits of generating each identifier as the top document. If you need more information on logits in transformers, [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) is a good entry point.

For more information about FIRST, refer to [Reddy et al. (2024)](https://arxiv.org/abs/2406.15657) if you are interested.

## Reranking with FirstMistral

[FirstMistral](https://arxiv.org/abs/2411.05508) is an LLM fine-tuned for listwise reranking using the FIRST approach.
Similar to RankZephyr, we will run an end-to-end multi-stage retrieval with FirstMistral.

If you are running on Colab, follow the same setup steps from the [RankZephyr lesson](./onboarding-rz.md#running-rankzephyr-on-a-colab-gpu) (Colab CLI, A100, the vLLM/CUDA 12.8 pins, and JDK 21). The environment is identical; only the run commands below change.

**Before running:** pre-seed the qrels as shown in the [RankZephyr lesson](./onboarding-rz.md) so evaluation doesn't hit Anserini's dead qrels download URL.

Assuming that necessary rank_llm installation steps to run RankZephyr have been performed, one can use the following commands to run FirstMistral on TREC DL19 and DL20:

---

#### DL19

```bash
rank-llm rerank \
  --model-path=castorini/first_mistral \
  --top-k-candidates=100 \
  --dataset=dl19 \
  --retrieval-method=SPLADE++_EnsembleDistil_ONNX \
  --prompt-template-path=src/rank_llm/rerank/prompt_templates/rank_zephyr_alpha_template.yaml \
  --context-size=4096 \
  --variable-passages \
  --use-logits \
  --use-alpha \
  --num-gpus 1
```

The results should be something like:

```text
Results:
ndcg_cut_10             all     0.7880
```

The command above performs first-stage retrieval with SPLADE to get the initial 100 candidates, followed by listwise reranking using FIRST with FirstMistral.

If you wish to compare FIRST's speed with traditional listwise reranking, omit the `--use-logits` and `--use-alpha` flags to perform traditional listwise reranking.

---

#### DL20

Now run the same pipeline on DL20:

```bash
rank-llm rerank \
  --model-path=castorini/first_mistral \
  --top-k-candidates=100 \
  --dataset=dl20 \
  --retrieval-method=SPLADE++_EnsembleDistil_ONNX \
  --prompt-template-path=src/rank_llm/rerank/prompt_templates/rank_zephyr_alpha_template.yaml \
  --context-size=4096 \
  --variable-passages \
  --use-logits \
  --use-alpha \
  --num-gpus 1
```

The results should be something like:

```text
Results:
ndcg_cut_10             all     0.7851
```

Running the same pipeline on DL20 gives a nDCG@10 score in a similar range, and now we have completed the FirstMistral experiments on both datasets.

---

That is all for this guide!
A reminder that this is just a gentle introduction, and the field is still largely an active area of research; we welcome you to join us in exploring the exciting possibilities of reranking with LLMs!

## Reproduction Log[*](https://github.com/castorini/pyserini/blob/master/docs/reproducibility.md)

The experiments in this guide could slightly vary in results due to the intrinsic randomness of LLMs, and particularly the `vLLM` library.
Thus, in addition to a log entry like the previous steps of the onboarding path, we also request that you add an entry to the table below indicating the precise numbers you obtained from running the experiments; we would like to keep track of these to better understand the variance from `vLLM`.
More specifically, we are interested in the `ndcg_cut_10` score for FirstMistral on both the TREC DL19 and DL20 datasets—the two experiments you have just completed.

| FirstMistral DL19 | Frequency |
|-------------------|-----------|
| 0.7880            | 1         |

| FirstMistral DL20 | Frequency |
|-------------------|-----------|
| 0.7906            | 1         |
| 0.7892            | 1         |
| 0.7889            | 1         |
| 0.7885            | 2         |
| 0.7877            | 1         |
| 0.7870            | 1         |
| 0.7865            | 1         |
| 0.7851            | 2         |
| 0.7843            | 5         |
| 0.7842            | 1         |
| 0.7829            | 1         |

For each result, if it is present in the corresponding table above, please increase its frequency by 1.
If a result is not present, add a new row (in sorted order) to the corresponding table with frequency 1.

After editing the tables above, add a log entry here as well like the previous guides:

+ Results reproduced by [@wu-ming233](https://github.com/wu-ming233) on 2025-01-08 (commit [`dac99f7`](https://github.com/castorini/rank_llm/commit/c908de0423747a3863ca288b072e4580b3a3adef))
+ Results reproduced by [@b8zhong](https://github.com/b8zhong) on 2025-02-03 (commit [`c908de0`](https://github.com/castorini/rank_llm/commit/c908de0423747a3863ca288b072e4580b3a3adef))
+ Results reproduced by [@vincent-4](https://github.com/vincent-4) on 2025-02-05 (commit [`4da0c46`](https://github.com/castorini/rank_llm/commit/4da0c46486fb31b65d41ec9a1fde7dacd9a05410))
+ Results reproduced by [@zdann15](https://github.com/zdann15) on 2025-02-12 (commit [`85302c2`](https://github.com/castorini/rank_llm/commit/85302c22c82c9008425651ead5b0c8e53b32cfe9))
+ Results reproduced by [@mithildamani256](https://github.com/mithildamani256) on 2025-02-15 (commit [`c91c011`](https://github.com/castorini/rank_llm/commit/c91c011ef5a60474144f9235551543d7fdd5c612))
+ Results reproduced by [@nihalmenon](https://github.com/nihalmenon) on 2025-02-19 (commit [`539c650`](https://github.com/castorini/rank_llm/commit/539c6502e42499e10a65c548f221b10b2e796296))
+ Results reproduced by [@lilyjge](https://github.com/lilyjge) on 2025-04-25 (commit [`b4ecd4c`](https://github.com/castorini/rank_llm/commit/b4ecd4c5512e95b7d00ca28c69149b13279fc274))
+ Results reproduced by [@Yaohui2019](https://github.com/Yaohui2019) on 2025-04-25 (commit [`d3a7a3c`](https://github.com/castorini/rank_llm/commit/d3a7a3c1690534b6f8f35c23a54e38321372d57d))
+ Results reproduced by [@Vik7am10](https://github.com/Vik7am10) on 2025-06-19 (commit [`baf3b39`](https://github.com/castorini/rank_llm/commit/baf3b39c06cb49c604960efcfa09aa83cfb0990c))
+ Results reproduced by [@kxwtan](https://github.com/kxwtan) on 2025-06-28 (commit [`e26abac`](https://github.com/castorini/rank_llm/commit/e26abaca5429ef4abc5e8d8f342c12e194fb230a))
+ Results reproduced by [@FarmersWrap](https://github.com/FarmersWrap) on 2025-09-20 (commit [`e84b530`](https://github.com/castorini/rank_llm/commit/e84b530d7c81cc6c5bc42cb8ca66932ac8c1a276))
+ Results reproduced by [@Raptors65](https://github.com/Raptors65) on 2025-11-01 (commit [`37faf8b`](https://github.com/castorini/rank_llm/commit/37faf8bf482b090ba90295d480dc8d324364d17f))
+ Results reproduced by [@ipouyall](https://github.com/ipouyall) on 2025-11-01 (commit [`b6ea7d58`](https://github.com/castorini/rank_llm/commit/b6ea7d587cd47174681d3580b6e1ea300d9ca55f))
+ Results reproduced by [@aaryanshroff](https://github.com/aaryanshroff) on 2025-11-01 (commit [`c4d06fe`](https://github.com/castorini/rank_llm/commit/c4d06fea82763ceb5223570eb5084f480d429003))
+ Results reproduced by [@nli33](https://github.com/nli33) on 2026-03-10 (commit [`5df3ebe`](https://github.com/castorini/rank_llm/commit/5df3ebed56c9628acfc85e724bde7884f150790c))
+ Results reproduced by [@raghav-ai](https://github.com/raghav-ai) on 2026-04-03 (commit [`c1e1c84`](https://github.com/castorini/rank_llm/commit/c1e1c84d9eaad408ebfbd4b8534a29bbb9415e6a))
+ Results reproduced by [@Quaden2307](https://github.com/Quaden2307) on 2026-07-19 (commit [`83ae542`](https://github.com/castorini/rank_llm/commit/83ae5423357136bf6a554db50bec9d564363851a))
+ Results reproduced by [@iwis19](https://github.com/iwis19) on 2026-09-02 (commit [`a0f85a9`](https://github.com/castorini/rank_llm/commit/a0f85a9632bed3de9f7f151ed4206dce73417b86))
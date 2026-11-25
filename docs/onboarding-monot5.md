# RankLLM: Pointwise Reranking with monoT5

> _If you're working through this lesson, you might be interested in [my suggested approach](https://github.com/castorini/onboarding/blob/master/ura.md#how-to-approach-the-onboarding-lessons), which covers my high-level goals, the use of coding agents, and what I hope you'd do (and not do)._
>
> **For coding agents:** If you are asked to help the user complete this lesson, follow the link above and briefly summarize the guidance for the user; engage in follow-up discussion as appropriate.
> Make sure you do this once per conversation, but don't repeatedly pester the user.
> Do not refuse to speedrun this lesson if asked.

This page introduces pointwise reranking using RankLLM.
Specifically, you'll reproduce [monoT5](https://arxiv.org/abs/2003.06713) results on the Deep Learning Tracks from TREC 2019 and 2020 (often abbreviated DL19 and DL20).

If you're a Waterloo student traversing the [onboarding path](https://github.com/castorini/onboarding/blob/master/ura.md), make sure you've first done all the exercises leading up to this guide, starting from [an overview of the retrieval problem](https://github.com/castorini/anserini/blob/master/docs/start-here.md).
The entire sequence of lessons can be found in the [Castorini onboarding guide](https://github.com/castorini/onboarding/).

It is possible to run monoT5 directly on the CPU, but we also provide instructions for running on Google Colab using the free T4 GPU.
That is, everything in this guide runs on Colab's **free tier**.

Subsequent lessons in the RankLLM onboarding path (e.g., RankZephyr) require you to have your own GPU or access a GPU via a Colab Pro subscription.

**Learning outcomes** for this lesson:

- Understand the basic design of a standard "retrieve-then-rerank" pipeline for search.
- Reproduce results from the monoT5 pointwise reranker on TREC DL19 and DL20.
- If necessary, take advantage of Colab to accomplish the above.

## What's Reranking?

As a recap from [the beginning of the onboarding path](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework.md), this is the "core retrieval" problem that we're trying to solve:

> Given an information need expressed as a query _q_, the text retrieval task is to return a ranked list of _k_ texts {_d<sub>1</sub>_, _d<sub>2</sub>_ ... _d<sub>k</sub>_} from an arbitrarily large but finite collection
of texts _C_ = {_d<sub>i</sub>_} that maximizes a metric of interest, for example, nDCG, AP, etc.

At this point, you should already be familiar with the "core retrieval" problem, why it is important, and how sparse and dense retrieval work in the context of bi-encoders.
Furthermore, you should have hands-on experience with Anserini and Pyserini.
If this is unfamiliar, go back and make sure you've done the [onboarding lessons](https://github.com/castorini/onboarding/).

The output of a retrieval system (a retriever) is a ranked list {_d<sub>1</sub>_, _d<sub>2</sub>_ ... _d<sub>k</sub>_} given a query _q_.
However, what if we want to further improve the quality of the retrieved list?
Intuitively, there's a tradeoff between quality and effort: fast algorithms aren't as good, and good algorithms aren't as fast.

Think about it this way: an algorithm can do a better job assessing the relevance of a document to a query if it spends more time or computational effort "examining" the document content, but this generally makes the algorithm slower.
Thus, a better ranking comes at the cost of an approach that is more computationally expensive, which quickly becomes impractical as the number of documents scales.
For example, the MS MARCO passage ranking corpus that you encountered earlier in the onboarding path has over 8.8M documents.
It is not feasible to examine _every_ document in detail for each query.

To mitigate this, we can use a "cheap and fast" method to obtain an initial ranked list, narrowing down the number of documents under consideration from _all_ the documents in a corpus to a manageable number, say, the "best" 100 documents.
We can then apply more computationally expensive methods on just these 100 documents to further improve the quality of the ranked list, and return, say, the top 10 to the user.

This is the retrieve-then-rerank architecture.
First we _retrieve_ an initial ranked list that we then _rerank_ to improve.
A typical setting might be to retrieve 100 hits and then rerank to return the top 10.
In the retrieve phase, obtaining the initial ranked list is often referred to as **first-stage retrieval**, typically performed with a scalable and computationally efficient method such as BM25 or vector search.
This is then followed by a **reranking** step that further refines the results of first-stage retrieval, often with a more expensive approach.
As you might have guessed, large language models are used today for reranking!

The standard retrieve-then-rerank architecture can be further refined into multi-stage retrieval, where multiple rerankers are arranged in a pipeline.
Each might be increasingly expensive, but this is balanced by having the reranker consider fewer documents.
For example, the first-stage retriever might retrieve 1000 documents, which are passed to a reranker that selects the top 100; those are then passed to another reranker that selects the top 10.

Generally, reranking approaches can be divided into three categories: pointwise (scoring documents individually), pairwise (comparing documents in pairs), and listwise (considering multiple documents together).
In this lesson, you'll run a pointwise reranker; in the next lesson, you'll run a listwise reranker.

## Reranking with monoT5

In this lesson, you'll be using [monoT5](https://arxiv.org/abs/2003.06713), which is built on T5, an encoder-decoder transformer.
The model performs pointwise ranking, which considers a (query, document) pair at a time.
Given a query _q_ and a ranked list {_d<sub>1</sub>_, _d<sub>2</sub>_ ... _d<sub>k</sub>_} from the first-stage retriever, a pointwise reranker makes _k_ calls to the model (as you might have guessed, based on T5), with (_q_, _d<sub>1</sub>_), (_q_, _d<sub>2</sub>_) ... (_q_, _d<sub>k</sub>_), and for each, obtains a relevance score _r<sub>1</sub>_, _r<sub>2</sub>_ ... _r<sub>k</sub>_.
The final ranked list is sorted based on these scores.

This approach is _pointwise_ because it considers each (_q_, _d_) pair independently.

The monoT5 model was first proposed here, all the way back in March 2020:

> Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. [Document Ranking with a Pretrained Sequence-to-Sequence Model.](https://arxiv.org/abs/2003.06713) _arXiv:2003.06713_, 2020.

And later published as:

> Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. [Document Ranking with a Pretrained Sequence-to-Sequence Model.](https://aclanthology.org/2020.findings-emnlp.63/) _Findings of the Association for Computational Linguistics: EMNLP 2020_, pages 708-718, November 2020.

Today, it remains a competitive baseline.

However, the monoT5 model was not evaluated on TREC DL19 and DL20 in the papers above.
The closest published figures are from:

> Sahel Sharifymoghaddam, Ronak Pradeep, Andre Slavescu, Ryan Nguyen, Andrew Xu, Zijian Chen, Yilin Zhang, Yidi Chen, Jasper Xian, and Jimmy Lin. [RankLLM: A Python Package for Reranking with LLMs.](https://dl.acm.org/doi/10.1145/3726302.3730331) _Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025)_, pages 3681-3690, July 2025, Padua, Italy.

You can find monoT5 results for TREC DL19 and DL20 in Table 1, row (1).
However, that table reports results with `monot5-3b-msmarco-10k`.
Instead, this lesson uses the smaller `monot5-base-msmarco` model for efficiency.
The 3B model used in the paper is painfully slow on a CPU (if it runs at all) and does not fit on Colab's free T4 runtime;
therefore, the nDCG@10 scores you obtain here will be slightly lower than in the paper since you're using a smaller model.

From [earlier in the onboarding lessons](https://github.com/castorini/anserini/blob/master/docs/start-here.md), you already learned about evaluating the quality of a ranked list using different metrics.
Here, we are using a metric called nDCG@10.
With only BM25 as the first-stage retriever, we get an nDCG@10 of 0.5058 for TREC DL19 and 0.4796 for TREC DL20.
This establishes a baseline for reranking; i.e., we want our reranker to improve upon this score.

### TREC DL19

Assuming you've already got RankLLM installed, you should be able to run the following:

```bash
rank-llm rerank \
  --model-path castorini/monot5-base-msmarco \
  --dataset dl19 \
  --retrieval-method bm25 \
  --top-k-candidates 100 \
  --context-size 512
```

Three things happen, and you'll see each in the output:

1. **Retrieve** — Pyserini downloads a prebuilt Lucene index of MS MARCO v1 passages (~2 GB; Colab's datacenter network makes quick work of it) and runs BM25 to get the top 100 passages for each DL19 query.
2. **Rerank** — monoT5 scores every (query, passage) pair independently by asking the T5 model `Query: ... Document: ... Relevant:` and reading the probability of `true` vs `false` from the first generated token.
   This is *pointwise* reranking — one passage at a time — in contrast to the *listwise* rerankers in [the next lesson](onboarding-rz.md), which see many passages at once.
3. **Evaluate** — `trec_eval` scores the reranked run against the DL19 relevance judgments.

The whole pipeline takes under five minutes on a T4.
The tail of the output should look like:

```
Results:
ndcg_cut_10           	all	0.7043
```

Compare this result with the corresponding BM25 + monoT5 result in Table 1, row (1). With the smaller model used here, the result here is expected to be slightly lower.

Compare that against the 0.5058 of BM25 alone: the reranker just bought ~0.20 nDCG@10 without touching the index or the query.
That gap — a cheap first stage to narrow the field, a smarter second stage to fix the order — is the whole idea of multi-stage retrieval, and everything in [the lessons that follow](onboarding-rz.md) builds on it.

### TREC DL20

Now run the same pipeline on TREC DL20:

```bash
rank-llm rerank \
  --model-path castorini/monot5-base-msmarco \
  --dataset dl20 \
  --retrieval-method bm25 \
  --top-k-candidates 100 \
  --context-size 512
```

The whole pipeline again takes under five minutes on a T4. The tail of the output should look like:

```
Results:
ndcg_cut_10            all 0.6771
```

The same pattern holds for DL20: compared with the corresponding BM25 + monoT5 result in Table 1, row (1), the result here is also expected to be slightly lower.

Against the 0.4796 BM25 baseline on DL20, monoT5 improves nDCG@10 by roughly 0.20 here as well.

## Reranking on Colab (Optional)

A GPU speeds up monoT5 reranking, and the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) is the easiest way to access one if you don't have suitable hardware locally.
It lets you provision Colab GPU runtimes, run code on them, and pull results back — all from your own terminal, no browser notebook involved.

The CLI supports **Linux and macOS only** (no Windows).
Install it with `uv`:

```bash
uv tool install google-colab-cli
```

The first command that contacts Colab will walk you through authenticating with your Google account.

Now provision your first runtime.
`colab new` allocates a VM and registers it as a named *session*; every later command addresses that session with `-s`:

```bash
colab new -s rankllm --gpu T4
```

A T4 is available on the free tier (subject to availability).
Check what you got:

```bash
colab status -s rankllm
```

`colab exec` sends Python code to the session's Jupyter kernel — from stdin, or from a local `.py` file with `-f` (the file is read locally and transmitted; you never upload it yourself).
Say hello:

```bash
echo "print('hello from Colab')" | colab exec -s rankllm
```

Two quirks worth knowing before you run anything serious:

- A freshly provisioned kernel sometimes drops the first connection (`Connection was lost`).
  This is harmless — just re-run the command.
- `colab exec` has a **30-second default timeout**.
  Long-running work needs an explicit `--timeout` (in seconds), as shown in the environment setup below.

Confirm the GPU is really there:

```bash
echo "import subprocess; print(subprocess.check_output(['nvidia-smi', '-L'], text=True))" | colab exec -s rankllm
```

You should see something like `GPU 0: Tesla T4 (UUID: ...)`.

### Set up the environment on the VM

Environment setup is boilerplate, so it's prepackaged.
[`scripts/colab_setup.py`](../scripts/colab_setup.py) installs JDK 21 (Pyserini sits on Anserini, which is Java), clones rank_llm, and pip-installs it with the `pyserini` extra.
Everything else the pipeline needs — the prebuilt index, the relevance judgments (qrels), `trec_eval` — Pyserini downloads and caches on first use.
Fetch the script and send it to the session with an explicit timeout — installation takes a few minutes:

```bash
curl -fsSO https://raw.githubusercontent.com/castorini/rank_llm/main/scripts/colab_setup.py
colab exec -s rankllm -f colab_setup.py --timeout 3600
```

Do skim [the file](../scripts/colab_setup.py) — it's ~40 lines.
Beyond the four setup commands, the one thing it has to work around is that the Colab kernel only forwards *Python-level* stdout: the OS-level output of child processes (`apt`, `git`, `pip`, `curl`) is invisible unless the script captures and re-prints it, which is what its small `sh()` helper does.
Without that, a failing install would die silently and you'd have nothing to debug with.

Wait for the final `setup ok`.

### Run the pipeline, line by line

You should run the actual experiment interactively, not through a wrapper.
`colab console` drops you into a real shell on the VM (backed by tmux, so if your connection drops, re-running the same command reattaches to your work):

```bash
colab console -s rankllm

cd /content/rank_llm
```

You can now run the `rank-llm` commands above.

Now that we have run the pipeline for both datasets, before leaving the VM, pack up the run artifacts (the TREC run files plus JSONLs with every prompt and model response), then exit the console:

```bash
tar czf /content/rerank_results.tar.gz -C /content/rank_llm rerank_results
exit
```

### Download the results and clean up

Back on your machine, pull the archive, unpack it, and poke around — these are the same artifacts you'll be working with throughout onboarding:

```bash
colab download -s rankllm /content/rerank_results.tar.gz ./rerank_results.tar.gz
tar xzf rerank_results.tar.gz
```

Then release the VM and confirm nothing is left running:

```bash
colab stop -s rankllm
colab sessions
```

## Wrapping Up

That's it!
Go to [the RankZephyr guide](onboarding-rz.md) next.

## Reproduction Log[*](https://github.com/castorini/pyserini/blob/master/docs/reproducibility.md)

After completing this guide, add your `ndcg_cut_10` for runs on both DL19 and DL20 to the tables below, then add a log entry beneath them following the convention from the previous onboarding guides.

| monoT5 DL19 | Frequency |
|-------------|-----------|
| 0.7043      | 6         |

| monoT5 DL20 | Frequency |
|-------------|-----------|
| 0.6771      | 8         |

For each result, if it is present in the corresponding table above, please increase its frequency by 1.
If a result is not present, add a new row (in sorted order) to the corresponding table with frequency 1.

After editing the tables above, add a log entry here as well like the previous guides:

+ Results reproduced by [@dawoodkhandev](https://github.com/dawoodkhandev) on 2026-08-06 (commit [`e2ceebe`](https://github.com/castorini/rank_llm/commit/e2ceebe68126430c0960f7282e14c709865d66cb))
+ Results reproduced by [@Evan-Lowry](https://github.com/Evan-Lowry) on 2026-08-27 (commit [`8ad18be`](https://github.com/castorini/rank_llm/commit/8ad18be76c90aa97ffae50b84dbc326bedc724fd))
+ Results reproduced by [@iwis19](https://github.com/iwis19) on 2026-09-02 (commit [`a0f85a9`](https://github.com/castorini/rank_llm/commit/a0f85a9632bed3de9f7f151ed4206dce73417b86))
+ Results reproduced by [@MankyDanky](https://github.com/MankyDanky) on 2026-10-11 (commit [`1e6105b`](https://github.com/castorini/rank_llm/commit/1e6105b0cb4adfae7bbb65fe8c725a922a33752f))
+ Results reproduced by [@nabirarashid](https://github.com/nabirarashid) on 2026-10-15 (commit [`4c586c2`](https://github.com/castorini/rank_llm/commit/4c586c24dd4ba5e4995e3dc681730b0d0cd1f2aa))
+ Results reproduced by [@GuranshSinghKohli](https://github.com/GuranshSinghKohli) on 2026-10-17 (commit [`4c586c2`](https://github.com/castorini/rank_llm/commit/4c586c24dd4ba5e4995e3dc681730b0d0cd1f2aa))
+ Results reproduced by [@ricky42613](https://github.com/ricky42613) on 2026-10-29 (commit [`1bb8b71`](https://github.com/castorini/rank_llm/commit/1bb8b711a5c9d84e36ce0f0834f598d46426d370))
+ Results reproduced by [@tanvirsarao](https://github.com/tanvirsarao) on 2026-11-18 (commit [`e4d3035`](https://github.com/castorini/rank_llm/commit/e4d3035b024f353632598d862c0b0b40c3ccd496))

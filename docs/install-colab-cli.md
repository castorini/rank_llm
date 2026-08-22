# RankLLM: Pointwise Reranking with monoT5 on Colab

The rank_llm step of the onboarding path requires a GPU, and the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) is the easiest way to get one if you don't have suitable hardware locally.
It lets you provision Colab GPU runtimes, run code on them, and pull results back — all from your own terminal, no browser notebook involved.

This guide has two parts:

1. Install the Colab CLI and sanity-check it.
2. Run your first multi-stage retrieval pipeline on a Colab GPU: BM25 retrieval over TREC DL20, reranked with [monoT5](https://arxiv.org/abs/2003.06713).

Everything in this guide runs on Colab's **free tier**.
The 7B listwise rerankers you'll meet in the [RankZephyr](onboarding-rz.md) and [FirstMistral](onboarding-first.md) lessons don't fit on the free T4 GPU — that's what the Colab Pro subscription discussed there is for — but the workflow you learn here is exactly the workflow you'll use to run them.

Work through this guide first, then continue with [onboarding-rz.md](onboarding-rz.md).
As with the previous guides in the onboarding path, don't just copy-paste your way through — each command below is one step of a remote-execution workflow, and the point is to understand what each step does.

**Learning outcomes** for this guide:

- Install and authenticate the Colab CLI.
- Understand the session lifecycle: `colab new` → `colab exec` / `colab console` → `colab download` → `colab stop`.
- Understand the quirks of driving a remote Jupyter kernel from a terminal (execution timeouts, streaming subprocess output).
- Run an end-to-end retrieve-then-rerank pipeline (BM25 → monoT5) on TREC DL20 and evaluate it with nDCG@10.

## Install and Sanity-Check the Colab CLI

The CLI supports **Linux and macOS only** (no Windows).
Install it with `uv` (or `pip install "google-colab-cli" "jupyter-kernel-client<1.0.0"` into any Python environment):

```bash
# jupyter-kernel-client 1.0.0 renamed KernelClient, which google-colab-cli 0.6.0
# still imports, so a default install crashes on the first `colab exec`. Pin
# below 1.0.0 until the CLI is updated upstream.
uv tool install google-colab-cli --with "jupyter-kernel-client<1.0.0"
```

The first command that contacts Colab will walk you through authenticating with your Google account.

Now provision your first runtime.
`colab new` allocates a VM and registers it as a named *session*; every later command addresses that session with `-s`:

```bash
colab new -s sanity --gpu T4
```

A T4 is available on the free tier (subject to availability).
Check what you got:

```bash
colab status -s sanity
```

`colab exec` sends Python code to the session's Jupyter kernel — from stdin, or from a local `.py` file with `-f` (the file is read locally and transmitted; you never upload it yourself).
Say hello:

```bash
echo "print('hello from Colab')" | colab exec -s sanity
```

Two quirks worth knowing before you run anything serious:

- A freshly provisioned kernel sometimes drops the first connection (`Connection was lost`).
  This is harmless — just re-run the command.
- `colab exec` has a **30-second default timeout**.
  Long-running work needs an explicit `--timeout` (in seconds), which you'll see in Part 2.

Confirm the GPU is really there:

```bash
echo "import subprocess; print(subprocess.check_output(['nvidia-smi', '-L'], text=True))" | colab exec -s sanity
```

You should see something like `GPU 0: Tesla T4 (UUID: ...)`.
That's the whole sanity check — release the VM:

```bash
colab stop -s sanity
```

## What's Pointwise Reranking?

As a recap from [here](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework.md), this is the "core retrieval" problem that we're trying to solve:

> Given an information need expressed as a query _q_, the text retrieval task is to return a ranked list of _k_ texts {_d<sub>1</sub>_, _d<sub>2</sub>_ ... _d<sub>k</sub>_} from an arbitrarily large but finite collection
of texts _C_ = {_d<sub>i</sub>_} that maximizes a metric of interest, for example, nDCG, AP, etc.

**Multi-Stage Retrieval.**
At this point of the onboarding path, we are already fairly familiar with the "core retrieval" problem above, and implemented sparse and dense retrieval in Pyserini that obtains such a ranked list {_d<sub>1</sub>_, _d<sub>2</sub>_ ... _d<sub>k</sub>_} given a query $q$.
However, what if we want to further improve the quality of the retrieved list?
Intuitively, to achieve a better ranking, the algorithms we run will also be more computationally expensive, which quickly becomes impractical as the number of documents scale (e.g. 8,841,823 documents in the MS MARCO passage ranking corpus).

To mitigate this, we can still proceed with the "cheaper" methods we've used before to get an "initial" ranked list, narrowing down the number of documents to a manageable number, say from 8,841,823 to 1000.
On top of this initial list, we can then apply more computationally expensive algorithms on just these 1000 documents to further improve the quality of the retrieved list.

This is the idea of **multi-stage retrieval**.
Obtaining the initial list is referred to as the **first-stage retrieval**, often done with a computationally efficient method, followed by a **reranking** step that further refines the results of first-stage retrieval, often with a more expensive approach.

**Reranking with Large Language Models.**
There are many ways we can leverage LLMs to perform reranking.
Generally, the approaches can be divided into three categories: pointwise (scoring documents individually), pairwise (comparing documents in pairs), and listwise (considering multiple documents together).
In this lesson, you'll run a pointwise reranker; the next lesson, you'll run a listwise reranker.

## Reranking with monoT5

You will now run a complete multi-stage retrieval pipeline on TREC DL20: first-stage retrieval with BM25 — the ranking function you know from the anserini and pyserini guides — followed by reranking the top 100 candidates per query with [monoT5](https://arxiv.org/abs/2003.06713), a pointwise T5 reranker from our group.
BM25 alone scores **0.4796** nDCG@10 on DL20; watch what the reranker does to that number.

(Why not RankZephyr here?
Its 7B weights alone are ~14 GB in fp16 — they barely fit on the free T4's 16 GB before you even allocate a KV cache.
The listwise experiments in the [RankZephyr](onboarding-rz.md) and [FirstMistral](onboarding-first.md) lessons therefore run on bigger GPUs via Colab Pro; monoT5-base is a couple hundred million parameters and runs comfortably, and the point of this guide is the workflow.)

Provision a fresh session:

```bash
colab new -s rankllm --gpu T4
```

### Set up the environment on the VM

Environment setup is boilerplate, so it's prepackaged.
[`scripts/colab_setup.py`](../scripts/colab_setup.py) installs JDK 21 (Pyserini sits on Anserini, which is Java), clones rank_llm, pip-installs it with the `pyserini` extra, and pre-seeds the DL20 relevance judgments (qrels) that the eval step below needs.
That last part works around a real bug: Anserini's own qrels download is a hardcoded URL that no longer resolves (the file moved to a different path upstream), so without this the pipeline gets all the way through retrieval and reranking and then fails at the very last step, evaluation.
The script fetches the file itself and drops it where Anserini's evaluator expects to find it already cached, so it never has to make that broken request.
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

The actual experiment you should run interactively, not through a wrapper.
`colab console` drops you into a real shell on the VM (backed by tmux, so if your connection drops, re-running the same command reattaches to your work):

```bash
colab console -s rankllm
```

Inside that shell, run the pipeline:

```bash
cd /content/rank_llm

# pyserini 1.2.0 constructs an OpenAI client at import time and raises if no
# key is set -- even though BM25 retrieval never calls OpenAI. Any placeholder
# satisfies the import; it is never actually used.
export OPENAI_API_KEY=sk-not-a-real-key

python src/rank_llm/scripts/run_rank_llm.py \
  --model_path=castorini/monot5-base-msmarco \
  --top_k_candidates=100 --dataset=dl20 \
  --retrieval_method=bm25 --context_size=512
```

Three things happen, and you'll see each in the output:

1. **Retrieve** — Pyserini downloads a prebuilt Lucene index of MS MARCO v1 passages (~2 GB; Colab's datacenter network makes quick work of it) and runs BM25 to get the top 100 passages for each DL20 query.
2. **Rerank** — monoT5 scores every (query, passage) pair independently by asking the T5 model `Query: ... Document: ... Relevant:` and reading the probability of `true` vs `false` from the first generated token.
   This is *pointwise* reranking — one passage at a time — in contrast to the *listwise* rerankers in [the next lesson](onboarding-rz.md), which see many passages at once.
3. **Evaluate** — `trec_eval` scores the reranked run against the DL20 relevance judgments.

The whole pipeline takes under five minutes on a T4.
The tail of the output should look like:

```
Results:
ndcg_cut_10           	all	0.6771
```

Compare that against the 0.4796 of BM25 alone: the reranker just bought ~0.20 nDCG@10 without touching the index or the query.
That gap — a cheap first stage to narrow the field, a smarter second stage to fix the order — is the whole idea of multi-stage retrieval, and everything in [the lessons that follow](onboarding-rz.md) builds on it.

Before leaving the VM, pack up the run artifacts (the TREC run file plus a JSONL with every prompt and model response), then exit the console:

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

Go to [the RankZephyr guide](onboarding-rz.md) next.

## Reproduction Log[*](https://github.com/castorini/pyserini/blob/master/docs/reproducibility.md)

After completing this guide, add your `ndcg_cut_10` to the table below, then add a log entry beneath it following the convention from the previous onboarding guides.

| monoT5 DL20 | Frequency |
|-------------|-----------|
| 0.6771      | 1         |

If your result is present in the table above, please increase its frequency by 1.
If your result is not present, add a new row (in sorted order) to the table with frequency 1.

After editing the table above, add a log entry here as well like the previous guides:

+ Results reproduced by [@dawoodkhandev](https://github.com/dawoodkhandev) on 2026-08-06 (commit [`e2ceebe`](https://github.com/castorini/rank_llm/commit/e2ceebe68126430c0960f7282e14c709865d66cb))

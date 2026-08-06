# Install the Colab CLI and Rerank on a Free Colab GPU

The rank_llm step of the onboarding path requires a GPU, and the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) is the easiest way to get one if you don't have suitable hardware locally.
It lets you provision Colab GPU runtimes, run code on them, and pull results back — all from your own terminal, no browser notebook involved.

This guide has two parts:

1. Install the Colab CLI and sanity-check it.
2. Run your first multi-stage retrieval pipeline on a Colab GPU: BM25 retrieval over TREC DL20, reranked with [monoT5](https://arxiv.org/abs/2003.06713).

Everything in this guide runs on Colab's **free tier**.
The 7B listwise rerankers you'll meet in [onboarding.md](onboarding.md) (RankZephyr, FirstMistral) don't fit on the free T4 GPU — that's what the Colab Pro subscription discussed there is for — but the workflow you learn here is exactly the workflow you'll use to run them.

Work through this guide first, then continue with [onboarding.md](onboarding.md).
As with the previous guides in the onboarding path, don't just copy-paste your way through — each command below is one step of a remote-execution workflow, and the point is to understand what each step does.

**Learning outcomes** for this guide:

- Install and authenticate the Colab CLI.
- Understand the session lifecycle: `colab new` → `colab exec` / `colab console` → `colab download` → `colab stop`.
- Understand the quirks of driving a remote Jupyter kernel from a terminal (execution timeouts, streaming subprocess output).
- Run an end-to-end retrieve-then-rerank pipeline (BM25 → monoT5) on TREC DL20 and evaluate it with nDCG@10.

## Part 1: Install and sanity-check the Colab CLI

The CLI supports **Linux and macOS only** (no Windows).
Install it with `uv` (or `pip install google-colab-cli` into any Python environment):

```bash
uv tool install google-colab-cli
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

## Part 2: Retrieve and rerank on a free Colab GPU

You will now run a complete multi-stage retrieval pipeline on TREC DL20: first-stage retrieval with BM25 — the ranking function you know from the anserini and pyserini guides — followed by reranking the top 100 candidates per query with [monoT5](https://arxiv.org/abs/2003.06713), a pointwise T5 reranker from our group.
BM25 alone scores **0.4796** nDCG@10 on DL20; watch what the reranker does to that number.

(Why not RankZephyr here?
Its 7B weights alone are ~14 GB in fp16 — they barely fit on the free T4's 16 GB before you even allocate a KV cache.
The listwise experiments in [onboarding.md](onboarding.md) therefore run on bigger GPUs via Colab Pro; monoT5-base is a couple hundred million parameters and runs comfortably, and the point of this guide is the workflow.)

Provision a fresh session:

```bash
colab new -s rankllm --gpu T4
```

### Set up the environment on the VM

Environment setup is boilerplate, so it's prepackaged.
[`scripts/colab_setup.py`](../scripts/colab_setup.py) installs JDK 21 (Pyserini sits on Anserini, which is Java), clones rank_llm, and pip-installs it with the `pyserini` extra.
Fetch it and send it to the session with an explicit timeout — installation takes a few minutes:

```bash
curl -fsSO https://raw.githubusercontent.com/castorini/rank_llm/main/scripts/colab_setup.py
colab exec -s rankllm -f colab_setup.py --timeout 3600
```

Do skim [the file](../scripts/colab_setup.py) — it's ~40 lines.
Beyond the three setup commands, the one thing it has to work around is that the Colab kernel only forwards *Python-level* stdout: the OS-level output of child processes (`apt`, `git`, `pip`) is invisible unless the script captures and re-prints it, which is what its small `sh()` helper does.
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
   This is *pointwise* reranking — one passage at a time — in contrast to the *listwise* rerankers in [onboarding.md](onboarding.md), which see many passages at once.
3. **Evaluate** — `trec_eval` scores the reranked run against the DL20 relevance judgments.

The whole pipeline takes under five minutes on a T4.
The tail of the output should look like:

```
Results:
ndcg_cut_10           	all	0.6771
```

Compare that against the 0.4796 of BM25 alone: the reranker just bought ~0.20 nDCG@10 without touching the index or the query.
That gap — a cheap first stage to narrow the field, a smarter second stage to fix the order — is the whole idea of multi-stage retrieval, and everything in [onboarding.md](onboarding.md) builds on it.

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

### Running the onboarding experiments on Colab

Once you have Colab Pro (see the note in [onboarding.md](onboarding.md)), the RankZephyr and FirstMistral experiments run the exact same way — provision a bigger GPU, same setup file, then the onboarding commands in the console:

```bash
colab new -s rankllm --gpu A100
colab exec -s rankllm -f colab_setup.py --timeout 3600
colab console -s rankllm
```

The listwise rerankers additionally need vLLM and the ONNX runtime (for the SPLADE++ first stage used there), so inside the console run

```bash
python -m pip install -q -e '/content/rank_llm[vllm]' onnxruntime
```

and then the `run_rank_llm.py` commands exactly as [onboarding.md](onboarding.md) gives them.

## Reproduction Log[*](https://github.com/castorini/pyserini/blob/master/docs/reproducibility.md)

After completing this guide, add an entry below following the convention from the previous onboarding guides, and include the `ndcg_cut_10` you obtained and the GPU you used.

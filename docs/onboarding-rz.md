# RankLLM: Listwise Reranking with RankZephyr

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
- Be able to run end-to-end multi-stage retrieval pipeline with RankZephyr and FirstMistral.

In this guide, we will focus on the listwise approach.
For more information about pointwise and pairwise, one can refer to [Zhuang et al. (2024)](https://arxiv.org/abs/2310.14122) and [Qin et al. (2024)](https://arxiv.org/abs/2306.17563).

> Note: as you can tell from the years of the citations, reranking with LLMs is quite a recent topic; indeed, this is still a highly active area of research. Thus, beware that the "knowledge-cutoff" of this guide is Jan 2025.

## Understanding Listwise Reranking

Listwise reranking is very straightforward in intuition: present the LLM with the query and candidate documents, and ask the LLM to give an ordering of the documents.

The conceptual idea is not far from directly asking the LLM with a prompt like this:

```text
I will provide you with 3 passages, each indicated by a numerical identifier [].
Rank the passages based on their relevance to the search query: what is the most influential band of all time?

[1] The electric guitar's transformation from rhythm instrument to lead voice began with pioneering blues players, but truly exploded in the mid-1960s. Innovative playing techniques - from feedback to distortion to wah-wah pedals - became fundamental building blocks of rock music.

[2] The Beatles revolutionized popular music during the 1960s, setting unprecedented records with 20 Billboard #1 hits. Their innovative studio techniques in albums like Sgt. Pepper's transformed music production forever.

[3] A hair band (also called a ponytail holder, hairkeeper) is a styling aid used to fasten hair, particularly long hair, away from areas such as the face.

Order the passages from most to least relevant using their identifiers.
```

and the LLM responds with a ranking:

```text
[2] > [1] > [3]
```

The actual implementation has a little more nuance, but nothing complicated.

## The Sliding Window Approach

However, there's a practical challenge: LLMs have a context length limitation (typically 4096 tokens).
Unlike the toy example above, in practice we have many, much longer documents to rank.
Consequently, we can't feed everything to the model at once; this is where the sliding window approach comes in.

Let's understand this with a concrete example.
Suppose we have 10 documents `[A, B, C, D, E, F, G, H, I, J]` to rank, where the true ranking we want to obtain is `[J, I, H, G, F, E, D, C, B, A]`.
Suppose we cannot fit all 10 documents into the context length, and decide to use the sliding window approach with **window size** 5 and **stride** 3.
Sliding window will proceed from the back of the list to the front, scanning 5 documents at a time, and advancing the window by 3 documents at a time:

### Iteration 1

Our current ranking is: `[A, B, C, D, E, F, G, H, I, J]`.
The LLM examines the documents `[F, G, H, I, J]` and orders them as `[J, I, H, G, F]`.
Our ranking becomes: `[A, B, C, D, E, J, I, H, G, F]`.

### Iteration 2

Our current ranking is: `[A, B, C, D, E, J, I, H, G, F]`.
The LLM examines the documents `[C, D, E, J, I]` and orders them as `[J, I, E, D, C]`.
Our ranking becomes: `[A, B, J, I, E, D, C, H, G, F]`.

### Iteration 3

Our current ranking is: `[A, B, J, I, E, D, C, H, G, F]`.
The LLM examines the documents `[A, B, J, I, E]` and orders them as `[J, I, E, B, A]`.
Our ranking becomes: `[J, I, E, B, A, D, C, H, G, F]`.
This completes the sliding window process.

Notice how the final ranking is not exactly the same as the true ranking, but the top documents (the ones that are closer to the beginning of the list) are close to the true ranking.

In practice, we often use a window size of 20 and a stride of 10.

> Note: sliding window is not the only way to do listwise reranking with context length limitations. Alternative methods are being investigated as an active area of research.

## Reranking with RankZephyr

[RankZephyr](https://huggingface.co/castorini/rank_zephyr_7b_v1_full) is an LLM specifically fine-tuned for listwise reranking, led by [Pradeep et. al (2023)](https://arxiv.org/abs/2312.02724) at the University of Waterloo.
We will run end-to-end multi-stage retrieval pipeline with RankZephyr, realizing the listwise reranking with sliding window mechanism as described above.
Note that this will require a GPU with **at least 16GB of VRAM**.

If you are short of GPUs, we recommend purchasing a [Google Colab Pro](https://colab.research.google.com/) for $13.99 CAD.
> Why do we make you pay $13.99?
Many discussions and arguments were made internally to come to this decision.
This onboarding guide differs from previous ones in that it requires GPU resources, specifically due to working with LLMs.
Everybody is interested in LLMs these days, and for a reason; a considerable amount of research at the group currently are involved with LLMs.
However, they come with substantial computational requirements.
We could have designed a simpler, GPU-free exercise that runs on free Colab, but that would defeat the purpose of these onboarding paths - to give you hands-on experience with actual research work rather than toy examples.
While Castorini has GPU resources, we cannot practically provide access to everyone who starts the onboarding process, as it involves significant administrative overhead with university compute managers, and many students ultimately don't continue past the initial weeks.
Therefore, we ask you to invest $13.99 (3 cups of coffee :coffee:) in a Colab Pro subscription.
This investment not only enables you to complete the onboarding but also positions you for more interesting tasks should you join the group, as you'll have the necessary compute resources at your disposal.

### Installing rank_llm

Please refer to the [instructions here](https://github.com/castorini/rank_llm?tab=readme-ov-file#-instructions) to install rank_llm.

#### Running RankZephyr on a Colab GPU

If you don't have a local GPU, run this step on a Colab Pro **A100** runtime, driven from your own terminal with the [Colab CLI](https://github.com/googlecolab/google-colab-cli). Work through the steps in order.

First, install the Colab CLI (Linux/macOS only). The first command that contacts Colab walks you through a one-time Google sign-in:

```bash
uv tool install google-colab-cli --with "jupyter-kernel-client<1.0.0"
```

Then provision an A100 and open a shell on it:

```bash
colab new -s rankllm --gpu A100
colab console -s rankllm
```

This session stays open through several long steps — installs, then loading and compiling a 7B model — so it's worth knowing what a dropped connection means before you hit one:

- `Connection closed` on its own just means the client disconnected; the VM and your work are still there. Re-run `colab console -s rankllm` to reattach — it's tmux-backed.
- `Session 'rankllm' appears to be lost (404/401)` means the VM itself is gone. There's nothing to reattach to; reprovision with `colab new` and start over.
- To cut down on drops, keep your laptop from sleeping for the duration — on macOS: `caffeinate -i colab console -s rankllm` (this only holds sleep off while `colab console` is running; it exits, and normal sleep resumes, the moment that does — whether that's `exit`, Ctrl-C, or a dropped connection — so there's nothing to remember to turn off).

Inside that shell, clone rank_llm and install it with the vLLM and Pyserini extras:

```bash
git clone https://github.com/castorini/rank_llm.git && cd rank_llm
pip install -e '.[vllm,pyserini]' onnxruntime
```

Next, pin a CUDA 12.8-matched vLLM. The current Colab image ships CUDA 12.8, but a default vLLM is built for CUDA 13 and crashes on import with `libcudart.so.13`. Pinning vllm 0.11.0 + torch 2.8.0 from the cu128 index fixes it, and `transformers` must stay on the 4.55.x line that this vLLM targets:

```bash
pip install --force-reinstall "vllm==0.11.0" "torch==2.8.0" --extra-index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.55.4"
```

Install JDK 21, which the Anserini/Pyserini first-stage retrieval needs:

```bash
apt-get install -y openjdk-21-jdk-headless
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

Setup is done. Stay in this console session for the actual run below — don't stop the runtime yet.

#### Running the RankZephyr Model

We can run the RankZephyr model with the command:

> **Before running:** the evaluation step downloads relevance judgments (qrels) that Anserini currently fetches from a hardcoded URL that no longer resolves (the qrels were moved to a `qrels/` folder upstream). Pre-seed the qrels into Pyserini's cache first so evaluation doesn't fail:
>
> ```bash
> mkdir -p ~/.cache/pyserini/topics-and-qrels
> curl -sSL -o ~/.cache/pyserini/topics-and-qrels/qrels.dl20-passage.txt \
>   https://raw.githubusercontent.com/castorini/anserini-tools/master/qrels/qrels.dl20-passage.txt
> ```

```bash
rank-llm rerank \
  --model-path=castorini/rank_zephyr_7b_v1_full \
  --top-k-candidates=100 \
  --dataset=dl20 \
  --retrieval-method=SPLADE++_EnsembleDistil_ONNX \
  --prompt-template-path=src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml  --context-size=4096 --variable-passages
```

The results should be something like:

```text
Results:
ndcg_cut_10             all     0.8201
```

Note that the result you get may vary slightly with the number above.

_Where is the first-stage retrieval?_
It is hidden in the `--retrieval-method=SPLADE++_EnsembleDistil_ONNX` flag.
We are using the [SPLADE](https://www.pinecone.io/learn/splade/) model as our sparse first-stage retriever, retrieving the top 100 candidates, followed by the RankZephyr model to rerank these 100 candidates.

Before leaving the VM, exit the console:

```bash
exit
```

Back on your machine, stop the runtime so you stop spending compute units:

```bash
colab stop -s rankllm
```

That is all for this guide!
A reminder that this is just a gentle introduction, and the field is still largely an active area of research; we welcome you to join us in exploring the exciting possibilities of reranking with LLMs!

Go to [the lesson on the FIRST model](./onboarding-first.md) next.

## Reproduction Log[*](https://github.com/castorini/pyserini/blob/master/docs/reproducibility.md)

The experiments in this guide could slightly vary in results due to the intrinsic randomness of LLMs, and particularly the `vLLM` library.
Thus, in addition to a log entry like the previous steps of the onboarding path, we also request that you add an entry to the table below indicating the precise numbers you obtained from running the experiments; we would like to keep track of these to better understand the variance from `vLLM`.
More specifically, we are interested in the `ndcg_cut_10` score for the RankZephyr and FirstMistral models on the DL20 dataset–the two experiments you have just completed.

| RankZephyr DL20 | Frequency |
|-----------------|-----------|
| 0.8201          | 2         |
| 0.8199          | 1         |
| 0.8198          | 3         |
| 0.8197          | 5         |
| 0.8169          | 1         |
| 0.8162          | 1         |
| 0.8151          | 1         |
| 0.8144          | 2         |

If your result is present in the table above, please increase its frequency by 1.
If your result is not present, add a new row (in sorted order) to the table with frequency 1.

After editing the table above, add a log entry here as well like the previous guides:

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

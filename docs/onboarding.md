# RankLLM: Reranking with LLMs

This guide presents a gentle introduction to reranking, the role it plays in the multi-stage retrieval pipeline, and how LLMs are emerging as a powerful tool for reranking.

If you're a Waterloo student traversing the [onboarding path](https://github.com/lintool/guide/blob/master/ura.md),
make sure you've first done all the exercises leading up to this guide, starting [here](https://github.com/castorini/anserini/blob/master/docs/start-here.md).
The [previous step](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework2.md) in the onboarding path is to implement sparse and dense retrieval in Pyserini.
In general, don't try to rush through this guide by just blindly copying and pasting commands into a shell;
that's what I call [cargo culting](https://en.wikipedia.org/wiki/Cargo_cult_programming).
Instead, really try to understand what's going on.

**Learning outcomes** for this guide:

- Understand the motivation and architecture of multi-stage retrieval.
- Understand listwise reranking with LLMs, using sliding windows.
- Understand how FIRST speeds up the reranking process by leveraging logits.
- Be able to run end-to-end multi-stage retrieval pipeline with RankZephyr and FirstMistral.

## Recap

As a recap from [here](https://github.com/castorini/pyserini/blob/master/docs/conceptual-framework.md), this is the "core retrieval" problem that we're trying to solve:

> Given an information need expressed as a query _q_, the text retrieval task is to return a ranked list of _k_ texts {_d<sub>1</sub>_, _d<sub>2</sub>_ ... _d<sub>k</sub>_} from an arbitrarily large but finite collection
of texts _C_ = {_d<sub>i</sub>_} that maximizes a metric of interest, for example, nDCG, AP, etc.

## Multi-stage retrieval

At this point of the onboarding path, we are already fairly familiar with the "core retrieval" problem above, and implemented sparse and dense retrieval in Pyserini that obtains such a ranked list {_d<sub>1</sub>_, _d<sub>2</sub>_ ... _d<sub>k</sub>_} given a query $q$. However, what if we want to further improve the quality of the retrieved list? Intuitively, to achieve a better ranking, the algorithms we run will also be more computationally expensive, which quickly becomes impractical as the number of documents scale (e.g. 8,841,823 documents in the MS MARCO passage ranking corpus).

To mitigate this, we can still proceed with the "cheaper" methods we've used before to get an "initial" ranked list, narrowing down the number of documents to a manageable number, say from 8,841,823 to 1000. On top of this initial list, we can then apply more computationally expensive algorithms on just these 1000 documents to further improve the quality of the retrieved list.

This is the idea of **multi-stage retrieval**. Obtaining the initial list is referred to as the **first-stage retrieval**, often done with a computationally efficient method, followed by a **reranking** step that further refines the results of first-stage retrieval, often with a more expensive approach.

## Reranking with Large Language Models

There are many ways we can leverage LLMs to perform reranking. Generally, the approaches can be divided into three categories: pointwise (scoring documents individually), pairwise (comparing documents in pairs), and listwise (considering multiple documents together). In this guide, we will focus on the listwise approach. For more information about pointwise and pairwise, one can refer to [Zhuang et al. (2024)](https://arxiv.org/abs/2310.14122) and [Qin et al. (2024)](https://arxiv.org/abs/2306.17563).

> Note: as you can tell from the years of the citations, reranking with LLMs is quite a recent topic; indeed, this is still a highly active area of research. Thus, beware that the "knowledge-cutoff" of this guide is Jan 2025. 

### Understanding Listwise Reranking

Listwise reranking is very straightforward in intuition: present the LLM with the query and candidate documents, and ask the LLM to give an ordering of the documents.

The conceptual idea is not far from directly asking the LLM with a prompt like this:
```
I will provide you with 3 passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: what is the most influential band of all time?

[1] The electric guitar's transformation from rhythm instrument to lead voice began with pioneering blues players, but truly exploded in the mid-1960s. Innovative playing techniques - from feedback to distortion to wah-wah pedals - became fundamental building blocks of rock music.

[2] The Beatles revolutionized popular music during the 1960s, setting unprecedented records with 20 Billboard #1 hits. Their innovative studio techniques in albums like Sgt. Pepper's transformed music production forever.

[3] A hair band (also called a ponytail holder, hairkeeper) is a styling aid used to fasten hair, particularly long hair, away from areas such as the face.

Order the passages from most to least relevant using their identifiers.
```
and the LLM responds with a ranking:
```
[2] > [1] > [3]
```
The actual implementation has a little more nuance, but nothing complicated.

### The Sliding Window Approach

However, there's a practical challenge: LLMs have a context length limitation (typically 4096 tokens). Unlike the toy example above, in practice we have many, much longer documents to rank. Consequently, we can't feed everything to the model at once; this is where the sliding window approach comes in.

Let's understand this with a concrete example. Suppose we have 10 documents `[A, B, C, D, E, F, G, H, I, J]` to rank, where the true ranking we want to obtain is `[J, I, H, G, F, E, D, C, B, A]`. Suppose we cannot fit all 10 documents into the context length, and decide to use the sliding window approach with **window size** 5 and **step size** 3. Sliding window will proceed from the back of the list to the front, scanning 5 documents at a time, and advancing the window by 3 documents at a time:

#### Iteration 1
Our current ranking is: `[A, B, C, D, E, F, G, H, I, J]`. The LLM examines the documents `[F, G, H, I, J]` and orders them as `[J, I, H, G, F]`. Our ranking becomes: `[A, B, C, D, E, J, I, H, G, F]`.

#### Iteration 2
Our current ranking is: `[A, B, C, D, E, J, I, H, G, F]`. The LLM examines the documents `[C, D, E, J, I]` and orders them as `[J, I, E, D, C]`. Our ranking becomes: `[A, B, J, I, E, D, C, H, G, F]`.

#### Iteration 3
Our current ranking is: `[A, B, J, I, E, D, C, H, G, F]`. The LLM examines the documents `[A, B, J, I, E]` and orders them as `[J, I, E, B, A]`. Our ranking becomes: `[J, I, E, B, A, D, C, H, G, F]`. This completes the sliding window process.

Notice how the final ranking is not exactly the same as the true ranking, but the top documents (the ones that are closer to the beginning of the list) are close to the true ranking.

In practice, we often use a window size of 20 and a step size of 10.

> Note: sliding window is not the only way to do listwise reranking with context length limitations. Alternative methods are being investigated as an active area of research.

### Reranking with RankZephyr

[RankZephyr](https://huggingface.co/castorini/rank_zephyr_7b_v1_full) is an LLM specifically fine-tuned for listwise reranking, led by [Pradeep et. al (2023)](https://arxiv.org/abs/2312.02724) at the University of Waterloo. We will run end-to-end multi-stage retrieval pipeline with RankZephyr, realizing the listwise reranking with sliding window mechanism as described above. Note that this will require a GPU with **at least 16GB of VRAM**.
> If you are short of GPUs, we recommend purchasing a [Google Colab Pro](https://colab.research.google.com/) for $13.99 CAD.

#### ❗ JDK 21 Warning

As rank_llm relies on [anserini](https://github.com/castorini/anserini), it is required that you have JDK 21 installed. Please note that using JDK 11 is not supported and may lead to errors for the following steps.

#### Create Conda Environment

```bash
conda create -n rankllm python=3.10
conda activate rankllm
```

#### Install Pytorch with CUDA (Windows/Linux)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Install Pytorch with MPS (Mac)
```bash
pip3 install torch torchvision torchaudio
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Install vLLM (Optional for faster inference)
```bash
pip install -e .[vllm]      # local installation for development
pip install rank-llm[vllm]  # or pip installation
```

#### Running the RankZephyr Model
We can run the RankZephyr model with the following command:
```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=dl20 \
--retrieval_method=SPLADE++_EnsembleDistil_ONNX --prompt_mode=rank_GPT  --context_size=4096 --variable_passages
```
The results should be something like:
```
Results:
ndcg_cut_10             all     0.8201
```

Including the `--vllm_batched` flag will allow you to run the model in batched mode using the `vLLM` library. Note that the result you get may vary slightly with the number above. 

_Where is the first-stage retrieval?_ It is hidden in the `--retrieval_method=SPLADE++_EnsembleDistil_ONNX` flag. We are using the [SPLADE](https://www.pinecone.io/learn/splade/) model as our sparse first-stage retriever, retrieving the top 100 candidates, followed by the RankZephyr model to rerank these 100 candidates.

## FIRST: First-token Reranking

FIRST (Faster Improved Listwise Reranking with Single Token Decoding) is a novel approach to reranking with LLMs that is up to 42% faster to inference than the "traditional" approach we have presented above for RankZephyr.

At a high level, instead of prompting the LLM to generate a full ranking of the documents (e.g. "[3] > [1] > [2]"), we examine the probability that each document will be ranked as the top document by the LLM, and infer the ranking from these probabilities. For example, if the probabilities of ranking documents 1, 2, 3 as the top document are 0.2, 0.1, 0.7, respectively, then we hypothesize that the true ranking is [3] > [1] > [2] without having to wait for the LLM to generate the full ranking in text, avoiding a major bottleneck in inference efficiency.
> How do we obtain such probabilities? We use the logits of generating each identifier as the top document. If you need more information on logits in transformers, [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) is a good entry point.

For more information about FIRST, refer to [Reddy et al. (2024)](https://arxiv.org/abs/2406.15657) if you are interested.

### Reranking with FirstMistral
[FirstMistral](https://arxiv.org/abs/2411.05508) is an LLM fine-tuned for listwise reranking using the FIRST approach. Similar to RankZephyr, we will run an end-to-end multi-stage retrieval with FirstMistral.

Assuming that necessary steps to run RankZephyr have been performed, one can follow the steps below to run FirstMistral:

#### Install vLLM (Mandatory for FirstMistral)
```bash
pip install -e .[vllm]      # local installation for development
pip install rank-llm[vllm]  # or pip installation
```

#### Run end to end - FirstMistral
```
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/first_mistral --top_k_candidates=100 --dataset=dl20 --retrieval_method=SPLADE++_EnsembleDistil_ONNX --prompt_mode=rank_GPT  --context_size=4096 --variable_passages --use_logits --use_alpha --vllm_batched --num_gpus 1
```
The results should be something like:
```
Results:
ndcg_cut_10             all     0.7851
```

This above performs first-stage retrieval with SPLADE to get the initial 100 candidates, followed by listwise reranking using FIRST with FirstMistral.

If you wish to compare FIRST's speed with traditional listwise reranking, omit the `--use_logits` and `--use_alpha` flags to perform traditional listwise reranking.

That is all for this guide! A reminder that this is just a gentle introduction, and the field is still largely an active area of research; we welcome you to join us in exploring the exciting possibilities of reranking with LLMs!

## Reproduction Log[*](https://github.com/castorini/pyserini/blob/master/docs/reproducibility.md)
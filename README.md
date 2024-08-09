# RankLLM

[![PyPI](https://img.shields.io/pypi/v/rank-llm?color=brightgreen)](https://pypi.org/project/rank-llm/)
[![Downloads](https://static.pepy.tech/personalized-badge/rank-llm?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/rank-llm)
[![Downloads](https://static.pepy.tech/personalized-badge/rank-llm?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/week)](https://pepy.tech/project/rank-llm)
[![Generic badge](https://img.shields.io/badge/arXiv-2309.15088-red.svg)](https://arxiv.org/abs/2309.15088)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)


We offer a suite of prompt-decoders, albeit with focus on open source LLMs compatible with [FastChat](https://github.com/lm-sys/FastChat?tab=readme-ov-file#supported-models) (e.g., Vicuna, Zephyr, etc.). Some of the code in this repository is borrowed from [RankGPT](https://github.com/sunnweiwei/RankGPT)!

# Releases
current_version = 0.12.8

## 📟 Instructions

### Create Conda Environment

```bash
conda create -n rankllm python=3.10
conda activate rankllm
```

### Install Pytorch with CUDA
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install openjdk with maven if you want to use the retriever
```bash
conda install -c conda-forge openjdk=21 maven -y
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run end to end - RankZephyr

We can run the RankZephyr model with the following command:
```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=dl20 \
--retrieval_method=SPLADE++_EnsembleDistil_ONNX --prompt_mode=rank_GPT  --context_size=4096 --variable_passages
```

Including the `--vllm_batched` flag will allow you to run the model in batched mode using the `vllm` library.
If you want to run multiple passes of the model, you can use the `--num_passes` flag.

### Run end to end - RankGPT4-o

We can run the RankGPT4-o model with the following command:
```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=gpt-4o --top_k_candidates=100 --dataset=dl20 \
  --retrieval_method=bm25 --prompt_mode=rank_GPT_APEER  --context_size=4096 --use_azure_openai
```
Note that the `--prompt_mode` is set to `rank_GPT_APEER` to use the LLM refined prompt from [APEER](https://arxiv.org/abs/2406.14449).
This can be changed to `rank_GPT` to use the original prompt.

If you would like to contribute to the project, please refer to the [contribution guidelines](CONTRIBUTING.md).

## Run end to end - LiT5

We can run the LiT5-Distill V2 model (which could rerank 100 documents in a single pass) with the following command:

```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/LiT5-Distill-large-v2 --top_k_candidates=100 --dataset=dl19 \
    --retrieval_method=bm25 --prompt_mode=LiT5  --context_size=150 --vllm_batched --batch_size=4 \
    --variable_passages --window_size=100
```

We can run the LiT5-Distill original model (which works with a window size of 20) with the following command:

```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/LiT5-Distill-large --top_k_candidates=100 --dataset=dl19 \
    --retrieval_method=bm25 --prompt_mode=LiT5  --context_size=150 --vllm_batched --batch_size=32 \
    --variable_passages
```

We can run the LiT5-Score model with the following command:

```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/LiT5-Score-large --top_k_candidates=100 --dataset=dl19 \
    --retrieval_method=bm25 --prompt_mode=LiT5 --context_size=150 --vllm_batched --batch_size=8 \
    --window_size=100 --variable_passages
```

## 🦙🐧 Model Zoo

The following is a table of our models hosted on HuggingFace:

| Model Name        | Hugging Face Identifier/Link                            |
|-------------------|---------------------------------------------|
| RankZephyr 7B V1 - Full - BF16      | [castorini/rank_zephyr_7b_v1_full](https://huggingface.co/castorini/rank_zephyr_7b_v1_full)               |
| RankVicuna 7B - V1      | [castorini/rank_vicuna_7b_v1](https://huggingface.co/castorini/rank_vicuna_7b_v1)               |
| RankVicuna 7B - V1 - No Data Augmentation    | [castorini/rank_vicuna_7b_v1_noda](https://huggingface.co/castorini/rank_vicuna_7b_v1_noda)               |
| RankVicuna 7B - V1 - FP16      | [castorini/rank_vicuna_7b_v1_fp16](https://huggingface.co/castorini/rank_vicuna_7b_v1_fp16)               |
| RankVicuna 7B - V1 - No Data Augmentation - FP16   | [castorini/rank_vicuna_7b_v1_noda_fp16](https://huggingface.co/castorini/rank_vicuna_7b_v1_noda_fp16)               |

### Model Zoo for LiT5

The following is a table specifically for our LiT5 models hosted on HuggingFace:

| Model Name            | Hugging Face Identifier/Link                            |
|-----------------------|---------------------------------------------|
| LiT5 Distill base     | [castorini/LiT5-Distill-base](https://huggingface.co/castorini/LiT5-Distill-base)          |
| LiT5 Distill large    | [castorini/LiT5-Distill-large](https://huggingface.co/castorini/LiT5-Distill-large)        |
| LiT5 Distill xl       | [castorini/LiT5-Distill-xl](https://huggingface.co/castorini/LiT5-Distill-xl)              |
| LiT5 Distill base v2  | [castorini/LiT5-Distill-base-v2](https://huggingface.co/castorini/LiT5-Distill-base-v2)    |
| LiT5 Distill large v2 | [castorini/LiT5-Distill-large-v2](https://huggingface.co/castorini/LiT5-Distill-large-v2)  |
| LiT5 Distill xl v2    | [castorini/LiT5-Distill-xl-v2](https://huggingface.co/castorini/LiT5-Distill-xl-v2)        |
| LiT5 Score base       | [castorini/LiT5-Score-base](https://huggingface.co/castorini/LiT5-Score-base)              |
| LiT5 Score large      | [castorini/LiT5-Score-large](https://huggingface.co/castorini/LiT5-Score-large)            |
| LiT5 Score xl         | [castorini/LiT5-Score-xl](https://huggingface.co/castorini/LiT5-Score-xl)                  |

## ✨ References

If you use RankLLM, please cite the following relevant papers: 

[[2309.15088] RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models](https://arxiv.org/abs/2309.15088)

<!-- {% raw %} -->
```
@ARTICLE{pradeep2023rankvicuna,
  title   = {{RankVicuna}: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models},
  author  = {Ronak Pradeep and Sahel Sharifymoghaddam and Jimmy Lin},
  year    = {2023},
  journal = {arXiv:2309.15088}
}
```
<!-- {% endraw %} -->


[[2312.02724] RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!](https://arxiv.org/abs/2312.02724)

<!-- {% raw %} -->
```
@ARTICLE{pradeep2023rankzephyr,
  title   = {{RankZephyr}: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!},
  author  = {Ronak Pradeep and Sahel Sharifymoghaddam and Jimmy Lin},
  year    = {2023},
  journal = {arXiv:2312.02724}
}
```
<!-- {% endraw %} -->

If you use one of the LiT5 models please cite the following relevant paper:

[[2312.16098] Scaling Down, LiTting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder Models](https://arxiv.org/abs/2312.16098)

```
@ARTICLE{tamber2023scaling,
  title   = {Scaling Down, LiTting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder Models},
  author  = {Manveer Singh Tamber and Ronak Pradeep and Jimmy Lin},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2312.16098}
}
```

## 🙏 Acknowledgments

This research is supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.

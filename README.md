# RankLLM

[![PyPI](https://img.shields.io/pypi/v/rank-llm?color=brightgreen)](https://pypi.org/project/rank-llm/)
[![Downloads](https://static.pepy.tech/personalized-badge/rank-llm?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/rank-llm)
[![Downloads](https://static.pepy.tech/personalized-badge/rank-llm?period=week&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads/week)](https://pepy.tech/project/rank-llm)
[![Generic badge](https://img.shields.io/badge/arXiv-2309.15088-red.svg)](https://arxiv.org/abs/2309.15088)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)


We offer a suite of rerankers - pointwise models like monoT5 and listwise models with a focus on open source LLMs compatible with [FastChat](https://github.com/lm-sys/FastChat?tab=readme-ov-file#supported-models) (e.g., Vicuna, Zephyr, etc.), [vLLM](https://https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang) or [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). We also support RankGPT variants, which are proprietary listwise rerankers. Addtionally, we support reranking with the first-token logits only to improve inference efficiency.  Some of the code in this repository is borrowed from [RankGPT](https://github.com/sunnweiwei/RankGPT), [PyGaggle](https://github.com/castorini/pygaggle), and [LiT5](https://github.com/castorini/LiT5)!

# Releases
current_version = 0.20.3

**Note for Mac Users:** RankLLM is not compatible with Apple Silicon (M1/M2) chips. However, you can still run it by using the Intel-based version of Anaconda and launching your terminal through Rosetta 2.

## 📟 Instructions

### Create Conda Environment

```bash
conda create -n rankllm python=3.10
conda activate rankllm
```

### Install Pytorch with CUDA (Windows/Linux)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install Pytorch with MPS (Mac)
```bash
pip3 install torch torchvision torchaudio
```

### Install openjdk with maven if you want to use the retriever
```bash
conda install -c conda-forge openjdk=21 maven -y
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

If building `nmslib` failed during installation, try manually installing the library with `conda install -c conda-forge nmslib` and following it up with `pip install -r requirements.txt` again.

### Install vLLM or SGLang (Optional)

#### vLLM

```bash
pip install rank-llm[vllm]  # pip installation
pip install -e .[vllm]      # or local installation for development
```

#### SGLang

```bash
pip install rank-llm[sglang]  # pip installation
pip install -e .[sglang]      # or local installation for development
```

Remember to install flashinfer to use `SGLang` backend.

```bash
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

#### TensorRT-LLM

Ensure [OpenMIP](https://www.open-mpi.org/software/ompi/v5.0/) is installed

```bash
pip install rank-llm[tensorrt-llm]  # pip installation
pip install -e .[tensorrt-llm]      # or local installation for development
```

### Run end to end - RankZephyr

We can run the RankZephyr model with the following command:
```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=dl20 \
--retrieval_method=SPLADE++_EnsembleDistil_ONNX --prompt_mode=rank_GPT  --context_size=4096 --variable_passages
```

Including the `--vllm_batched` flag will allow you to run the model in batched mode using the `vLLM` library.

Including the `--sglang_batched` flag will allow you to run the model in batched mode using the `SGLang` library.

Including the `--tensorrt_batched` flag will allow you to run the model in batched mode using the `TensorRT-LLM` library.

If you want to run multiple passes of the model, you can use the `--num_passes` flag.

### Run end to end - RankGPT4-o

We can run the RankGPT4-o model with the following command:
```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=gpt-4o --top_k_candidates=100 --dataset=dl20 \
  --retrieval_method=bm25 --prompt_mode=rank_GPT_APEER  --context_size=4096 --use_azure_openai
```
Note that the `--prompt_mode` is set to `rank_GPT_APEER` to use the LLM refined prompt from [APEER](https://arxiv.org/abs/2406.14449).
This can be changed to `rank_GPT` to use the original prompt.

### Run end to end - LiT5

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

### Run end to end - monoT5

The following runs the 3B variant of monoT5 trained for 10K steps:

```
python src/rank_llm/scripts/run_rank_llm.py --model_path=castorini/monot5-3b-msmarco-10k --top_k_candidates=1000 --dataset=dl19 \
  --retrieval_method=bm25 --prompt_mode=monot5 --context_size=512
```

Note that we usually rerank 1K candidates with monoT5.

### Run end to end - FirstMistral

We can run the FirstMistral model, reranking using the first-token logits only with the following command:

```
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/first_mistral --top_k_candidates=100 --dataset=dl20 --retrieval_method=SPLADE++_EnsembleDistil_ONNX --prompt_mode=rank_GPT  --context_size=4096 --variable_passages --use_logits --use_alpha --vllm_batched --num_gpus 1
```

Omit `--use_logits` if you wish to perform traditional listwise reranking.

If you would like to contribute to the project, please refer to the [contribution guidelines](CONTRIBUTING.md).

## 🦙🐧 Model Zoo

The following is a table of the listwise models our repository was primarily built to handle (with the models hosted on HuggingFace):

`vLLM`, `SGLang`, and `TensorRT-LLM` backends are only supported for `RankZephyr` and `RankVicuna` models.

| Model Name        | Hugging Face Identifier/Link                            |
|-------------------|---------------------------------------------|
| RankZephyr 7B V1 - Full - BF16      | [castorini/rank_zephyr_7b_v1_full](https://huggingface.co/castorini/rank_zephyr_7b_v1_full)               |
| RankVicuna 7B - V1      | [castorini/rank_vicuna_7b_v1](https://huggingface.co/castorini/rank_vicuna_7b_v1)               |
| RankVicuna 7B - V1 - No Data Augmentation    | [castorini/rank_vicuna_7b_v1_noda](https://huggingface.co/castorini/rank_vicuna_7b_v1_noda)               |
| RankVicuna 7B - V1 - FP16      | [castorini/rank_vicuna_7b_v1_fp16](https://huggingface.co/castorini/rank_vicuna_7b_v1_fp16)               |
| RankVicuna 7B - V1 - No Data Augmentation - FP16   | [castorini/rank_vicuna_7b_v1_noda_fp16](https://huggingface.co/castorini/rank_vicuna_7b_v1_noda_fp16)               |

We also officially support the following rerankers built by our group:

### LiT5 Suite

The following is a table specifically for our LiT5 suite of models hosted on HuggingFace:

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

Now you can run top-100 reranking with the v2 model in a single pass while maintaining efficiency!

### monoT5 Suite - Pointwise Rerankers

The following is a table specifically for our monoT5 suite of models hosted on HuggingFace:

| Model Name                        | Hugging Face Identifier/Link                            |
|-----------------------------------|--------------------------------------------------------|
| monoT5 Small MSMARCO 10K          | [castorini/monot5-small-msmarco-10k](https://huggingface.co/castorini/monot5-small-msmarco-10k)       |
| monoT5 Small MSMARCO 100K         | [castorini/monot5-small-msmarco-100k](https://huggingface.co/castorini/monot5-small-msmarco-100k)     |
| monoT5 Base MSMARCO               | [castorini/monot5-base-msmarco](https://huggingface.co/castorini/monot5-base-msmarco)                 |
| monoT5 Base MSMARCO 10K           | [castorini/monot5-base-msmarco-10k](https://huggingface.co/castorini/monot5-base-msmarco-10k)         |
| monoT5 Large MSMARCO 10K          | [castorini/monot5-large-msmarco-10k](https://huggingface.co/castorini/monot5-large-msmarco-10k)       |
| monoT5 Large MSMARCO              | [castorini/monot5-large-msmarco](https://huggingface.co/castorini/monot5-large-msmarco)               |
| monoT5 3B MSMARCO 10K             | [castorini/monot5-3b-msmarco-10k](https://huggingface.co/castorini/monot5-3b-msmarco-10k)             |
| monoT5 3B MSMARCO                 | [castorini/monot5-3b-msmarco](https://huggingface.co/castorini/monot5-3b-msmarco)                     |
| monoT5 Base Med MSMARCO           | [castorini/monot5-base-med-msmarco](https://huggingface.co/castorini/monot5-base-med-msmarco)         |
| monoT5 3B Med MSMARCO             | [castorini/monot5-3b-med-msmarco](https://huggingface.co/castorini/monot5-3b-med-msmarco)             |

We recommend the Med models for biomedical retrieval. We also provide both 10K (generally better OOD effectiveness) and 100K checkpoints (better in-domain).

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
  journal = {arXiv:2312.16098}
}
```

If you use one of the monoT5 models please cite the following relevant paper:

[[2101.05667] The Expando-Mono-Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence Models](https://arxiv.org/abs/2101.05667)

```
@ARTICLE{pradeep2021emd,
  title = {The Expando-Mono-Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence Models},
  author = {Ronak Pradeep and Rodrigo Nogueira and Jimmy Lin},
  year = {2021},
  journal = {arXiv:2101.05667},
}
```

If you use the FirstMistral model, please consider citing:

[[2411.05508] An Early FIRST Reproduction and Improvements to Single-Token Decoding for Fast Listwise Reranking](https://arxiv.org/abs/2411.05508)

```
@ARTICLE{chen2024firstrepro,
  title   = title={An Early FIRST Reproduction and Improvements to Single-Token Decoding for Fast Listwise Reranking},
  author  = {Zijian Chen and Ronak Pradeep and Jimmy Lin},
  year    = {2024},
  journal = {arXiv:2411.05508}
}
```

If you would like to cite the FIRST methodology, please consider citing:

[[2406.15657] FIRST: Faster Improved Listwise Reranking with Single Token Decoding](https://arxiv.org/abs/2406.15657)

```
@ARTICLE{reddy2024first,
  title   = {FIRST: Faster Improved Listwise Reranking with Single Token Decoding},
  author  = {Reddy, Revanth Gangi and Doo, JaeHyeok and Xu, Yifei and Sultan, Md Arafat and Swain, Deevya and Sil, Avirup and Ji, Heng},
  year    = {2024}
  journal = {arXiv:2406.15657},
}
```

## 🙏 Acknowledgments

This research is supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.

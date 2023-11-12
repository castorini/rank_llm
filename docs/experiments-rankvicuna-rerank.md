# RankLLM: RankVicuna Re-rank Collections

This guide provides instructions to reproduce the RankVicuna Re-rank model described in the following paper.  

Note, we recommend using >24GB RAM for top-20 ranking, and >30GB RAM for top-100 ranking experiments.  
Additionally, we recommend using A100 with high-RAM option when working on Colab.  
The expected runtime for the experiments is around 4 mins for top-20 dl19, 3 mins for top-20 dl20, 45 mins for top-100 dl19, and 30 mins for top-100 dl20.

> Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. [RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models.](https://arxiv.org/abs/2309.15088) *arXiv:2309.15088.*

## dl19/dl20 Dataset Ranking

Summary of results:
| Condition                            | ndcg_cut_10 |
|:-------------------------------------|------------:|
| dl19 (Top 20)                        |    0.6164   |
| dl19 (Top 100)                       |    0.6664   |
| dl20 (Top 20)                        |    0.5985   |
| dl20 (Top 100)                       |    0.6540   |

To install:
```bash
$ git clone https://github.com/castorini/rank_llm
$ cd rank_llm/
$ pip install -r requirements.txt
```

Additionally, run the following command to reduce RAM usage:
```bash
$ export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
```

dl19 Top-20 Ranking:
```bash
$ python rank_llm/run_rank_llm.py --model_path={"castorini/rank_vicuna_7b_v1"} --dataset={"dl19"} --prompt_mode={"rank_GPT"} --retrieval_method={"bm25"} --top_k_candidates={20};
```

dl19 Top-100 Ranking:
```bash
$ python rank_llm/run_rank_llm.py --model_path={"castorini/rank_vicuna_7b_v1"} --dataset={"dl19"} --prompt_mode={"rank_GPT"} --retrieval_method={"bm25"} --top_k_candidates={100};
```

dl20 Top-20 Ranking:
```bash
$ python rank_llm/run_rank_llm.py --model_path={"castorini/rank_vicuna_7b_v1"} --dataset={"dl20"} --prompt_mode={"rank_GPT"} --retrieval_method={"bm25"} --top_k_candidates={20};
```

dl20 Top-100 Ranking:
```bash
$ python rank_llm/run_rank_llm.py --model_path={"castorini/rank_vicuna_7b_v1"} --dataset={"dl20"} --prompt_mode={"rank_GPT"} --retrieval_method={"bm25"} --top_k_candidates={100};
```

Additionally, you can experiment with demos for RankVicuna's two types of inline dataset: query-documents format and query-hits format.

demo/rerank_demo_docs.py Ranking:

```bash
# query-documents
# https://github.com/castorini/rank_llm/blob/main/demo/rerank_demo_docs.py
$ python demo/rerank_demo_docs.py
```

Expected output: `[3] > [1] > [13] > [10] > [11] > [15] > [4] > [14] > [16] > [2] > [12] > [17] > [18] > [5] > [6] > [7] > [8] > [9] > [19] > [20]`  

Note: the outputs of the rerank_demo_docs.py is only relevant upto the first 4 indices, given that there are 4 passages in the experiment.

demo/rerank_demo_hits.py Ranking:

```bash
# query-hits
# https://github.com/castorini/rank_llm/blob/main/demo/rerank_demo_hits.py
$ python demo/rerank_demo_hits.py
```

Expected output: `[1] > [2] > [4] > [7] > [8] > [3] > [6] > [5] > [9] > [10] > [11] > [12] > [13] > [14] > [15] > [16] > [17] > [18] > [19] > [20]`

Note: the outputs of the rerank_demo_hits.py is only relevant upto the first 8 indices, given that there are 8 passages in the experiment.
import argparse
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM

"""
Note: You need to run the vllm server with the following command:
```bash
RANK_MODEL_ID="openai/gpt-oss-20b"
RANK_PORT=48003
RANK_VLLM_LOG="vllm_server_48003.log"
CUDA_VISIBLE_DEVICES=0 vllm serve "$RANK_MODEL_ID"  \
 --port "$RANK_PORT" \
 --dtype auto \
 --gpu-memory-utilization 0.9 \
 --max-model-len 32768 \
 --enable-prompt-tokens-details \
 --enable-prefix-caching \
 --reasoning-parser openai_gptoss \
 > "$RANK_VLLM_LOG" 2>&1 &
```

Change the model id to `liuwenhan/reasonrank-32B` or `Qwen/Qwen3-8B` to use the reasonrank or qwen3 model, respectively.
The reasoning parser for the qwen3 model is `qwen3`, exclude the flag for the reasonrank model.

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--context_size", type=int, default=4096 * 4)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--reasoning_token_budget", type=int, default=4096 * 4 - 10)
    parser.add_argument("--max_passage_words", type=int, default=800)
    parser.add_argument("--prompt_template_path", type=str, default=None)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "aops",
            "biology",
            "earth-science",
            "economics",
            "leetcode",
            "pony",
            "psychology",
            "robotics",
            "stackoverflow",
            "sustainable-living",
            "theoremqa-questions",
            "theoremqa-theorems",
        ],
    )
    args = parser.parse_args()

    TEMPLATES = files("rank_llm.rerank.prompt_templates")
    for retrieve_model in ["bge", "splade", "diver", "reason_embed"]:
        for task in args.tasks:
            # Use the solo retrieval results for diver and reason_embed and naf for bge and splade
            if retrieve_model in ["diver", "reason_embed"]:
                file_name = f"../bright_fusion/reranker_requests_sigir_submission/first_stage/{retrieve_model}/retrieve_results_bright-{task}-{retrieve_model}_top100.jsonl"
            else:
                file_name = f"../bright_fusion/reranker_requests_sigir_submission/naf/bm25qs_{retrieve_model}/retrieve_results_bright-{task}-naf-bm25qs-{retrieve_model}_top100.jsonl"
            path = Path(f"../bright_fusion/rerank_results/{args.model.split('/')[-1]}")
            path.mkdir(parents=True, exist_ok=True)
            model_name = retrieve_model
            if os.path.exists(
                os.path.join(path, f"{task}-{model_name}_top100_metrics.json")
            ):
                print(f"Skipping {task}-{model_name} because it already exists")
                continue
            print(f"Reading retrieve results from {file_name}")
            retrieve_results = read_requests_from_file(file_name)
            qrels = f"../bright_fusion/qrels/qrels.bright-{task}.txt"
            retrieve_ndcg_10 = EvalFunction.from_results(retrieve_results, qrels)
            if args.prompt_template_path is None:
                prompt_file_name = f"rank_zephyr_alpha_template_{task}.yaml"
                if "reasonrank" in args.model:
                    prompt_file_name = "reasonrank_template.yaml"
                prompt_file_path = TEMPLATES / prompt_file_name
            else:
                prompt_file_path = args.prompt_template_path
            reranker = Reranker(
                RankListwiseOSLLM(
                    context_size=args.context_size,
                    model=args.model,
                    use_alpha=False if "reasonrank" in args.model else True,
                    is_thinking=True,
                    reasoning_token_budget=args.reasoning_token_budget,
                    base_url=f"http://localhost:{args.port}/v1",
                    max_passage_words=args.max_passage_words,
                    prompt_template_path=prompt_file_path,
                )
            )
            kwargs = {"populate_invocations_history": True}
            rerank_results = reranker.rerank_batch(requests=retrieve_results, **kwargs)
            rerank_ndcg_10 = EvalFunction.from_results(rerank_results, qrels)
            writer = DataWriter(rerank_results)

            writer.write_in_jsonl_format(
                os.path.join(path, f"{task}-{model_name}_top100.jsonl")
            )
            writer.write_in_trec_eval_format(
                os.path.join(path, f"{task}-{model_name}_top100.txt")
            )
            writer.write_inference_invocations_history(
                os.path.join(path, f"{task}-{model_name}_top100_invocations.json")
            )
            with open(
                os.path.join(path, f"{task}-{model_name}_top100_metrics.json"), "w"
            ) as f:
                json.dump({"retrieve": retrieve_ndcg_10, "rerank": rerank_ndcg_10}, f)


if __name__ == "__main__":
    main()

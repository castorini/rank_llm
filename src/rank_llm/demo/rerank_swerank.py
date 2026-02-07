"""
SweRank benchmark reranker for SWE-bench and Loc-Bench datasets.
"""

from rank_llm.rerank.listwise import RankListwiseOSLLM
from rank_llm.rerank import Reranker
from rank_llm.data import DataWriter, read_requests_from_file
from importlib.resources import files
import json
import os
import glob
from pathlib import Path
import argparse
from enum import Enum
from rank_llm.rerank.listwise.rank_listwise_os_llm import RerankType
from rank_llm.utils import find_best_gpu

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class Dataset(str, Enum):
    LOC_BENCH = "loc-bench"
    SWE_BENCH_LITE = "swe-bench-lite"


DATASETS_DIR = "/home/aaryans/SweRank/datasets_local/datasets"
MODEL = "Salesforce/SweRankLLM-Small"


# From https://arxiv.org/pdf/2505.07849
WINDOW_SIZE = 10
STRIDE = 5
CONTEXT_SIZE = 16348


def get_local_id_map(datasets_dir: str, dataset_prefix: str):
    """Build mapping: Query Text -> Instance ID from local BEIR datasets."""
    id_map = {}
    pattern = os.path.join(
        datasets_dir, f"{dataset_prefix}-function_*", "queries.jsonl"
    )

    for qfile in glob.glob(pattern):
        with open(qfile, "r") as f:
            for line in f:
                data = json.loads(line)
                text = "".join(data.get("text", "").split()).lower()
                instance_id = data.get("_id")
                if text and instance_id:
                    id_map[text] = instance_id
    return id_map


def write_swerank_format(results, output_file: Path, id_map):
    """Write results in SweRank JSONL format."""
    with open(output_file, "w") as f:
        for result in results:
            query_text = "".join(result.query.text.split()).lower()
            instance_id = id_map.get(query_text, str(result.query.qid))
            docs = [cand.docid for cand in result.candidates]
            f.write(json.dumps({"instance_id": instance_id, "docs": docs}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Dataset,
        default=Dataset.LOC_BENCH,
        choices=list(Dataset),
    )
    parser.add_argument("--output_dir", type=str, default="swerank_results")
    parser.add_argument(
        "--gpu_memory",
        type=float,
        default=0.55,
        help="GPU memory utilization fraction (0.0 to 1.0)",
    )
    args = parser.parse_args()

    input_file = f"/store2/scratch/ura/aaryans/retrieval_data/retrieve_results_swerank_{args.dataset}_top100.jsonl"
    output_dir = args.output_dir
    dataset_prefix = args.dataset
    gpu_memory = args.gpu_memory

    os.environ["CUDA_VISIBLE_DEVICES"] = find_best_gpu(gpu_memory)

    print(f"Dataset: {args.dataset}")
    print(f"Input file: {input_file}")
    print(f"GPU Memory: {gpu_memory}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    requests = read_requests_from_file(input_file)

    # Rerank
    templates = files("rank_llm.rerank.prompt_templates")
    extra_kwargs = {}
    if args.dataset == Dataset.LOC_BENCH:
        extra_kwargs["batch_size"] = 1
    model_coordinator = RankListwiseOSLLM(
        model=MODEL,
        context_size=CONTEXT_SIZE,
        prompt_template_path=str(templates / "swerank_github_issue_template.yaml"),
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        gpu_memory_utilization=gpu_memory,
        rerank_type=RerankType.CODE,
        **extra_kwargs
    )
    reranker = Reranker(model_coordinator)
    rerank_results = reranker.rerank_batch(requests, populate_invocations_history=True)

    # Save results
    writer = DataWriter(rerank_results)
    writer.write_in_jsonl_format(output_path / "rerank_results.jsonl")
    writer.write_in_trec_eval_format(output_path / "rerank_results.txt")
    writer.write_inference_invocations_history(
        output_path / "inference_invocations_history.json"
    )

    # Write SweRank format
    id_map = get_local_id_map(DATASETS_DIR, dataset_prefix)
    swerank_output = output_path / "swerank_eval_results.jsonl"
    write_swerank_format(rerank_results, swerank_output, id_map)

    print(f"Results: {swerank_output}")

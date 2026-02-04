"""
SweRank reranker for GitHub issue code localization.

Usage:
    python rerank_swerank.py <input_file> [output_dir]
"""

import os
import sys

# Force spawn method to avoid "Cannot re-initialize CUDA in forked subprocess" error.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from importlib.resources import files
from pathlib import Path

from rank_llm.data import DataWriter, read_requests_from_file
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM


def main():
    if len(sys.argv) < 2:
        print("Usage: python rerank_swerank.py <input_file> [output_dir]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "swerank_results"

    # Load retrieval results
    print(f"Loading: {input_file}")
    requests = read_requests_from_file(input_file)
    print(f"Loaded {len(requests)} queries")

    # Setup SweRank reranker
    TEMPLATES = files("rank_llm.rerank.prompt_templates")
    model_coordinator = RankListwiseOSLLM(
        model="Salesforce/SweRankLLM-Small",  # 8B params, or use SweRankLLM-Large (33B)
        context_size=4096,
        prompt_template_path=str(TEMPLATES / "swerank_github_issue_template.yaml"),
        window_size=20,
    )
    reranker = Reranker(model_coordinator)

    # Rerank
    print("Reranking...")
    kwargs = {"populate_invocations_history": True}
    rerank_results = reranker.rerank_batch(requests, **kwargs)
    print(f"Reranked {len(rerank_results)} queries")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    writer = DataWriter(rerank_results)
    writer.write_in_jsonl_format(output_path / "rerank_results.jsonl")
    writer.write_in_trec_eval_format(output_path / "rerank_results.txt")
    writer.write_inference_invocations_history(
        output_path / "inference_invocations_history.json"
    )

    print(f"\nResults saved to: {output_path}")
    print(f"  - rerank_results.jsonl")
    print(f"  - rerank_results.txt")
    print(f"  - inference_invocations_history.json")


if __name__ == "__main__":
    main()

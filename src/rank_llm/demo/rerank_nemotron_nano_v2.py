"""
NVIDIA-Nemotron-Nano-9B-v2 reranker with vLLM
Runs both thinking and non-thinking modes back-to-back.
"""

from pathlib import Path
from importlib.resources import files

from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM
from rank_llm.retrieve import Retriever
from rank_llm.retrieve.topics_dict import TOPICS
from rank_llm.data import DataWriter

DATASET = "dl19"
TOP_K = 100

# retrieval only once (same for both modes)
requests = Retriever.from_dataset_with_prebuilt_index(DATASET, k=TOP_K)
topics = TOPICS[DATASET]
TEMPLATES = files("rank_llm.rerank.prompt_templates")

def run_mode(thinking: bool):
    """Run Nemotron reranker in either thinking or non-thinking mode."""
    mode_name = "thinking" if thinking else "nothink"
    template_file = "nemotron_thinking_template.yaml" if thinking else "nemotron_nonthinking_template.yaml"
    template_path = str(TEMPLATES / template_file)

    print(f"\n=== Running {mode_name} mode ===")
    print(f"Template: {template_file}")

    model_coordinator = RankListwiseOSLLM(
        model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        prompt_template_path=template_path,
    )
    reranker = Reranker(model_coordinator)

    # rerank all queries
    rerank_results = reranker.rerank_batch(
        requests,
        populate_invocations_history=True,
        top_k_retrieve=TOP_K,
    )

    # metrics
    bm25_score = EvalFunction.from_results(requests, topics)           # string
    rerank_score = EvalFunction.from_results(rerank_results, topics)   # string

    # Print EXACTLY like your original code (no numeric formatting)
    print(f"BM25 ndcg@10: {bm25_score}")
    print(f"Rerank ndcg@10: {rerank_score}")

    # Save results
    outdir = Path(f"demo_outputs/{DATASET}/nemotron_nano_v2_{mode_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    writer = DataWriter(rerank_results)
    writer.write_in_jsonl_format(outdir / "rerank_results.jsonl")
    writer.write_in_trec_eval_format(outdir / "rerank_results.txt")
    writer.write_inference_invocations_history(outdir / "inference_invocations_history.json")

    # Write eval results (compute improvement safely, but keep printed lines unchanged)
    with open(outdir / "eval_results.txt", "w") as f:
        f.write(f"BM25 ndcg@10: {bm25_score}\n")
        f.write(f"Rerank ndcg@10: {rerank_score}\n")


if __name__ == "__main__":
    print(f"Testing Nemotron Nano 9B v2 on {DATASET} (BM25 top-{TOP_K})")
    run_mode(thinking=True)
    run_mode(thinking=False)

import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.data import DataWriter
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker, get_genai_api_key, get_openai_api_key
from rank_llm.rerank.listwise import (
    RankListwiseOSLLM,
    SafeGenai,
    SafeOpenai,
    VicunaReranker,
    ZephyrReranker,
)
from rank_llm.rerank.listwise.lit5_reranker import LiT5DistillReranker
from rank_llm.rerank.pointwise.monot5 import MonoT5
from rank_llm.retrieve.retriever import Retriever
from rank_llm.retrieve.topics_dict import TOPICS

# create rerankers
monot5_reranker = Reranker(MonoT5("castorini/monot5-3b-msmarco-10k"))
v_reranker = VicunaReranker()
z_reranker = ZephyrReranker()
lit5_reranker = Reranker(LiT5DistillReranker("castorini/LiT5-Distill-large"))
mistral_reranker = Reranker(
    RankListwiseOSLLM(
        model="castorini/first_mistral",
        use_logits=True,
        use_alpha=True,
        vllm_batched=True,
    )
)
gpt_reranker = Reranker(SafeOpenai("gpt-4o-mini", 4096, keys=get_openai_api_key()))
gemini_reranker = Reranker(
    SafeGenai("gemini-2.0-flash-001", 4096, keys=get_genai_api_key())
)
rerankers = {
    "monot5": monot5_reranker,
    "rv": v_reranker,
    "rz": z_reranker,
    "lit5": lit5_reranker,
    "mistral": mistral_reranker,
}  # , "gpt": g_reranker, "gemini": gemini_reranker }
results = {}
for dataset in ["dl19", "dl20", "dl21", "dl22", "dl23"]:
    retrieved_results = Retriever.from_dataset_with_prebuilt_index(dataset, k=20)
    topics = TOPICS[dataset]
    ret_ndcg_10 = EvalFunction.from_results(retrieved_results, topics)
    for key, reranker in rerankers.items():
        kwargs = {"populate_invocations_history": True}
        rerank_results = reranker.rerank_batch(retrieved_results, **kwargs)
        # Save results
        writer = DataWriter(rerank_results)
        output_path_prefix = f"demo_outputs/{dataset}/{key}"
        Path(f"{output_path_prefix}").mkdir(parents=True, exist_ok=True)
        writer.write_in_jsonl_format(f"{output_path_prefix}/rerank_results.jsonl")
        writer.write_in_trec_eval_format(f"{output_path_prefix}/rerank_results.txt")
        writer.write_inference_invocations_history(
            f"{output_path_prefix}/inference_invocations_history.json"
        )
        # Eval
        rerank_ndcg_10 = EvalFunction.from_results(rerank_results, topics)
        # Response Analysis
        if key not in ["monot5", "duot5"]:
            use_alpha = True if key == "mistral" else False
            analyzer = ResponseAnalyzer.from_inline_results(
                rerank_results, use_alpha=use_alpha
            )
            error_counts = analyzer.count_errors()
        else:
            error_counts = {}
        results[(key, dataset)] = (ret_ndcg_10, rerank_ndcg_10, error_counts.__repr__())
        print("-----------\n")
        print(results)
        print("-----------\n")

with open("demo_outputs/aggeragate_open_source_results.json", "w") as f:
    json.dump(results, f)

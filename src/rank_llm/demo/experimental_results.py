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
from rank_llm.rerank import PromptMode, Reranker, get_genai_api_key, get_openai_api_key
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


def create_reranker(name: str):
    if name == "monot5":
        return Reranker(MonoT5("castorini/monot5-3b-msmarco-10k"))
    if name == "rv":
        return VicunaReranker()
    if name == "rz":
        return ZephyrReranker()
    if name == "lit5":
        return Reranker(LiT5DistillReranker("castorini/LiT5-Distill-large"))
    if name == "mistral":
        return Reranker(
            RankListwiseOSLLM(
                model="castorini/first_mistral",
                use_logits=True,
                use_alpha=True,
                vllm_batched=True,
            )
        )
    if name == "rank_gpt":
        return Reranker(
            SafeOpenai(
                "gpt-4o-mini",
                4096,
                prompt_mode=PromptMode.RANK_GPT,
                keys=get_openai_api_key(),
            )
        )
    if name == "lrl":
        return Reranker(
            SafeOpenai(
                "gpt-4o-mini",
                4096,
                prompt_mode=PromptMode.LRL,
                keys=get_openai_api_key(),
            )
        )
    if name == "rank_gpt_apeer":
        return Reranker(
            SafeOpenai(
                "gpt-4o-mini",
                4096,
                prompt_mode=PromptMode.RANK_GPT_APEER,
                keys=get_openai_api_key(),
            )
        )
    if name == "gemini":
        return Reranker(
            SafeGenai("gemini-2.0-flash-001", 4096, keys=get_genai_api_key())
        )


rerankers = [
    "monot5",
    "rv",
    "rz",
    "lit5",
    "mistral",
    "rank_gpt",
    "gemini",
    "lrl",
    "rank_gpt_apeer",
]
results = {}
for key in rerankers:
    reranker = create_reranker(key)
    for dataset in ["dl19", "dl20", "dl21", "dl22"]:  # , "dl23"
        retrieved_results = Retriever.from_dataset_with_prebuilt_index(dataset, k=100)
        topics = TOPICS[dataset]
        ret_ndcg_10 = EvalFunction.from_results(retrieved_results, topics)
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
        # TODO: For now skipping lrl and rank_gpt_apeer since the response analyzer does not support these prompt formats, yet.
        if key not in ["monot5", "duot5", "lrl", "rank_gpt_apeer"]:
            use_alpha = True if key == "mistral" else False
            analyzer = ResponseAnalyzer.from_inline_results(
                rerank_results, use_alpha=use_alpha
            )
            error_counts = analyzer.count_errors()
        else:
            error_counts = {}
        results[(key, dataset)] = (ret_ndcg_10, rerank_ndcg_10, error_counts.__repr__())
        with open(f"{output_path_prefix}/eval_results.txt", "w") as f:
            f.write(f"{(ret_ndcg_10, rerank_ndcg_10, error_counts.__repr__())}")

    # Free up the memory
    del reranker

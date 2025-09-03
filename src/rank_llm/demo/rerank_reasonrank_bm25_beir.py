import json
import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM
from rank_llm.retrieve.retriever import Retriever
from rank_llm.retrieve.topics_dict import TOPICS


def main():
    for dataset in ["scidocs", "dbpedia", "nfc", "covid", "news", "signal", "robust04"]:
        TEMPLATES = files("rank_llm.rerank.prompt_templates")
        retrieve_results = Retriever.from_dataset_with_prebuilt_index(dataset, k=100)
        qrels = TOPICS[dataset]
        retrieve_ndcg_10 = EvalFunction.from_results(retrieve_results, qrels)
        reranker = Reranker(
            RankListwiseOSLLM(
                context_size=4096 * 8,
                model="liuwenhan/reasonrank-32B",
                is_thinking=True,
                reasoning_token_budget=3072,
                window_size=20,
                stride=10,
                batch_size=1,
                num_gpus=1,
                prompt_template_path=(TEMPLATES / "reasonrank_template.yaml"),
            )
        )
        kwargs = {"populate_invocations_history": True}
        rerank_results = reranker.rerank_batch(requests=retrieve_results, **kwargs)
        del reranker
        rerank_ndcg_10 = EvalFunction.from_results(rerank_results, qrels)
        writer = DataWriter(rerank_results)
        path = Path(f"./rerank_results/reasonrank-32b")
        path.mkdir(parents=True, exist_ok=True)
        writer.write_in_jsonl_format(os.path.join(path, f"{dataset}_bm25_top100.jsonl"))
        writer.write_in_trec_eval_format(
            os.path.join(path, f"{dataset}_bm25_top100.txt")
        )
        writer.write_inference_invocations_history(
            os.path.join(path, f"{dataset}_bm25_top100_invocations.json")
        )
        with open(os.path.join(path, f"{dataset}_bm25_top100_metrics.json"), "w") as f:
            json.dump({"retrieve": retrieve_ndcg_10, "rerank": rerank_ndcg_10}, f)


if __name__ == "__main__":
    main()

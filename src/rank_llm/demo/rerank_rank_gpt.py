import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.rerank import Reranker, get_openai_api_key
from rank_llm.rerank.listwise import SafeOpenai
from rank_llm.retrieve import Retriever

# By default uses BM25 for retrieval
dataset_name = "dl19"
requests = Retriever.from_dataset_with_prebuilt_index(dataset_name)[:2]
model_coordinator = SafeOpenai(
    "gpt-4o-mini",
    4096,
    keys=get_openai_api_key(),
    prompt_template_path="src/rank_llm/rerank/prompt_templates/rank_gpt_template.yaml",
)
reranker = Reranker(model_coordinator)
kwargs = {"populate_invocations_history": True}
rerank_results = reranker.rerank_batch(requests, **kwargs)
print(rerank_results)

from pathlib import Path

from rank_llm.data import DataWriter

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_jsonl_format(f"demo_outputs/rerank_results.jsonl")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_inference_invocations_history(
    f"demo_outputs/inference_invocations_history.json"
)

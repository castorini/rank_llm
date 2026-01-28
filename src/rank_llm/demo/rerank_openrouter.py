import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter
from rank_llm.rerank import Reranker, get_openrouter_api_key
from rank_llm.rerank.listwise import SafeOpenai
from rank_llm.retrieve import Retriever

# By default uses BM25 for retrieval
dataset_name = "dl19"
requests = Retriever.from_dataset_with_prebuilt_index(dataset_name)
TEMPLATES = files("rank_llm.rerank.prompt_templates")
model_coordinator = SafeOpenai(
    "minimax/minimax-m2:free",
    4096,
    keys=get_openrouter_api_key(),
    base_url="https://openrouter.ai/api/v1/",
    api_type="openai",
    prompt_template_path=(TEMPLATES / "rank_zephyr_template.yaml"),
)
reranker = Reranker(model_coordinator)
kwargs = {"populate_invocations_history": True}
rerank_results = reranker.rerank_batch(requests, **kwargs)
print(rerank_results)

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_jsonl_format(f"demo_outputs/rerank_results.jsonl")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_inference_invocations_history(
    f"demo_outputs/inference_invocations_history.json"
)

import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter
from rank_llm.rerank import Reranker, get_genai_api_key
from rank_llm.rerank.listwise import SafeGenai
from rank_llm.retrieve import Retriever

# By default uses BM25 for retrieval
dataset_name = "dl19"
requests = Retriever.from_dataset_with_prebuilt_index(dataset_name)
TEMPLATES = files("rank_llm.rerank.prompt_templates")
model_coordinator = SafeGenai(
    "gemini-3-flash-preview",
    4096,
    keys=get_genai_api_key(),
    prompt_template_path=(TEMPLATES / "rank_zephyr_template.yaml"),
)
reranker = Reranker(model_coordinator)
kwargs = {"populate_invocations_history": True}
rerank_results = reranker.rerank_batch(requests, **kwargs)
print(rerank_results)

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_jsonl_format(f"demo_outputs/rerank_results_gemini.jsonl")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results_gemini.txt")
writer.write_inference_invocations_history(
    f"demo_outputs/inference_invocations_history_gemini.json"
)

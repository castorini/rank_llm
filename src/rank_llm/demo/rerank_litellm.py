"""Rerank using any LiteLLM-supported model.

Usage:
    pip install 'rank-llm[litellm]'
    export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, GROQ_API_KEY, etc.
    python -m rank_llm.demo.rerank_litellm
"""

import os
import sys
from importlib.resources import files
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import SafeLiteLLM
from rank_llm.retrieve import Retriever

dataset_name = "dl19"
requests = Retriever.from_dataset_with_prebuilt_index(dataset_name)
TEMPLATES = files("rank_llm.rerank.prompt_templates")

model_coordinator = SafeLiteLLM(
    model="openai/gpt-4o-mini",
    context_size=128000,
    prompt_mode=None,
    prompt_template_path=(TEMPLATES / "rank_zephyr_template.yaml"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

reranker = Reranker(model_coordinator)
kwargs = {"populate_invocations_history": True}
rerank_results = reranker.rerank_batch(requests, **kwargs)
print(rerank_results)

Path("demo_outputs/").mkdir(parents=True, exist_ok=True)
writer = DataWriter(rerank_results)
writer.write_in_jsonl_format("demo_outputs/rerank_results.jsonl")
writer.write_in_trec_eval_format("demo_outputs/rerank_results.txt")
writer.write_inference_invocations_history(
    "demo_outputs/inference_invocations_history.json"
)

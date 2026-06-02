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

TEMPLATES = files("rank_llm.rerank.prompt_templates")

# Configurable via environment variables
MODEL = os.environ.get("LITELLM_MODEL", "openai/gpt-4o-mini")
CONTEXT_SIZE = int(os.environ.get("LITELLM_CONTEXT_SIZE", "4096"))
DATASET = os.environ.get("LITELLM_DATASET", "dl19")
WINDOW_SIZE = int(os.environ.get("LITELLM_WINDOW_SIZE", "20"))
STRIDE = int(os.environ.get("LITELLM_STRIDE", "10"))
BATCH_SIZE = int(os.environ.get("LITELLM_BATCH_SIZE", "32"))
TEMPLATE = os.environ.get(
    "LITELLM_TEMPLATE", str(TEMPLATES / "rank_zephyr_template.yaml")
)

requests = Retriever.from_dataset_with_prebuilt_index(DATASET)

model_coordinator = SafeLiteLLM(
    model=MODEL,
    context_size=CONTEXT_SIZE,
    prompt_template_path=TEMPLATE,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    batch_size=BATCH_SIZE,
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

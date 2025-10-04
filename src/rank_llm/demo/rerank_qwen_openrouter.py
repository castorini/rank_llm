"""
Qwen-R1-Distill reranker using OpenRouter
Based on OpenRouter integration in commit 55d5c1f4a49080e92af1b19a184ea57982ae9f8b
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
from rank_llm.rerank import Reranker, get_openrouter_api_key
from rank_llm.rerank.listwise import SafeOpenai
from rank_llm.retrieve import Retriever

# By default uses BM25 for retrieval
dataset_name = "dl19"
requests = Retriever.from_dataset_with_prebuilt_index(dataset_name)
TEMPLATES = files("rank_llm.rerank.prompt_templates")

model = "deepseek/deepseek-r1-distill-qwen-14b:free"
print(f"Testing {len(requests)} queries")

# Test with thinking mode
try:
    thinking_coordinator = SafeOpenai(
        model, 4096, keys=get_openrouter_api_key(),
        base_url="https://openrouter.ai/api/v1/",
        prompt_template_path=(TEMPLATES / "qwen_thinking_template.yaml"),
    )
    thinking_reranker = Reranker(thinking_coordinator)
    print(f"Using {model} with thinking mode via OpenRouter")
    
    thinking_results = thinking_reranker.rerank_batch(requests, populate_invocations_history=True)
    
    # Save thinking results
    writer = DataWriter(thinking_results)
    Path("demo_outputs/").mkdir(parents=True, exist_ok=True)
    writer.write_in_jsonl_format("demo_outputs/qwen_openrouter_thinking_results.jsonl")
    writer.write_in_trec_eval_format("demo_outputs/qwen_openrouter_thinking_results.txt")
    writer.write_inference_invocations_history("demo_outputs/qwen_openrouter_thinking_invocations.json")
    
    print("Thinking mode results saved")
    
except Exception as e:
    print(f"Thinking mode failed: {e}")

# Test without thinking
try:
    non_thinking_coordinator = SafeOpenai(
        model, 4096, keys=get_openrouter_api_key(),
        base_url="https://openrouter.ai/api/v1/",
        prompt_template_path=(TEMPLATES / "qwen_non_thinking_template.yaml"),
    )
    non_thinking_reranker = Reranker(non_thinking_coordinator)
    print(f"Using {model} with non-thinking mode")
    
    non_thinking_results = non_thinking_reranker.rerank_batch(requests, populate_invocations_history=True)
    
    # Save non-thinking results
    writer = DataWriter(non_thinking_results)
    writer.write_in_jsonl_format("demo_outputs/qwen_openrouter_non_thinking_results.jsonl")
    writer.write_in_trec_eval_format("demo_outputs/qwen_openrouter_non_thinking_results.txt")
    writer.write_inference_invocations_history("demo_outputs/qwen_openrouter_non_thinking_invocations.json")
    
    print("Non-thinking mode results saved")
    
except Exception as e:
    print(f"Non-thinking mode failed: {e}")

print("Results saved to demo_outputs/")

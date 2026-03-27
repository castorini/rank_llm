# rank_llm Extras

Use the smallest dependency stack that supports the workflow in front of you.

## Default Contributor Install

```bash
uv sync --group dev --extra cloud --extra api
```

This covers:
- hosted OpenAI-compatible rerankers
- Gemini rerankers
- `rank-llm serve http`
- `rank-llm doctor`, `prompt`, `view`, `describe`, `schema`, and `validate`

## Optional Extras

| Workflow | `uv` install | `pip` install |
| --- | --- | --- |
| Hosted OpenAI or OpenRouter reranking only | `uv sync --group dev --extra openai` | `pip install -e ".[openai]"` |
| Hosted Gemini reranking only | `uv sync --group dev --extra genai` | `pip install -e ".[genai]"` |
| All hosted providers | `uv sync --group dev --extra cloud` | `pip install -e ".[cloud]"` |
| HTTP API server | `uv sync --group dev --extra api` | `pip install -e ".[api]"` |
| Retrieval or trec_eval-based evaluation | `uv sync --group dev --extra pyserini` | `pip install -e ".[pyserini]"` |
| Local Hugging Face and PyTorch rerankers | `uv sync --group dev --extra local` | `pip install -e ".[local]"` |
| vLLM batched inference | `uv sync --group dev --extra vllm` | `pip install -e ".[vllm]"` |
| SGLang batched inference | `uv sync --group dev --extra sglang` | `pip install -e ".[sglang]"` |
| TensorRT-LLM batched inference | `uv sync --group dev --extra tensorrt-llm` | `pip install -e ".[tensorrt-llm]"` |
| MCP server bundle | `uv sync --group dev --extra mcp` | `pip install -e ".[mcp]"` |
| Full HTTP and MCP server bundle | `uv sync --group dev --extra server` | `pip install -e ".[server]"` |
| Training scripts | `uv sync --group dev --extra training` | `pip install -e ".[training]"` |
| Everything | `uv sync --group dev --extra all` | `pip install -e ".[all]"` |

## Notes

- `genai` is the canonical Gemini extra. `gemini` exists as a compatibility alias.
- `cloud` aggregates the hosted-provider stacks.
- `api` is intentionally lighter than `server`.
- `mcp` currently pulls a larger bundle because it includes model-serving and `pyserini`-dependent pieces.
- Install `flashinfer` separately for some `sglang` setups and `flash-attn` separately for TensorRT-LLM or training workflows when needed.

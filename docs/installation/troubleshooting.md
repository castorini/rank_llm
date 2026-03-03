# Troubleshooting

## Check your setup quickly

```bash
python -m rank_llm.doctor
```

## Common issues

- `ImportError` for OpenAI classes:
  - Install: `pip install 'rank-llm[openai]'`
- `ImportError` for vLLM/listwise OSS classes:
  - Install: `pip install 'rank-llm[vllm]'`
- Hugging Face model download failures:
  - Verify internet access and Hugging Face availability.
- GPU unavailable:
  - Use `device="cpu"` for pointwise quickstart.

## If uv is missing

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

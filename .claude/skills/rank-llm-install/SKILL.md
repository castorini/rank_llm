# rank-llm-install

This skill is an installation/run orchestrator. Use the docs below as the source of truth.

## Default flow (minimal)

1. Follow `docs/installation/quickstart.md`.
2. Run `python -m rank_llm.doctor` if setup problems appear.

## Optional backends (only on explicit user request)

- OpenAI/OpenRouter: `docs/installation/extras.md`
- vLLM listwise: `docs/installation/extras.md`
- Gemini: `docs/installation/extras.md`
- Pyserini retriever: `docs/installation/extras.md`

Do not install `rank-llm[all]` unless user explicitly asks for it.

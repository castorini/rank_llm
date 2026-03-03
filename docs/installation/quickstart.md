# Quickstart (minimal install)

This path is intentionally minimal and targets Hugging Face pointwise reranking first.
Install extra backends only when needed.

## 1) Create an environment (uv-first)

```bash
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv venv
source .venv/bin/activate
```

## 2) Install RankLLM base package

```bash
uv pip install rank-llm
```

## 3) Run a minimal reranking script (pip-only)

```bash
python - <<'PY'
from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank import Reranker
from rank_llm.rerank.pointwise.monot5 import MonoT5

model = MonoT5(model="castorini/monot5-base-msmarco", device="cpu", batch_size=4)
reranker = Reranker(model)
request = Request(
    query=Query(text="what is bm25", qid="q1"),
    candidates=[
        Candidate(docid="d1", score=0.0, doc={"text": "BM25 is a bag-of-words ranking function used in search."}),
        Candidate(docid="d2", score=0.0, doc={"text": "Neural reranking uses cross-encoders to score query-document relevance."}),
        Candidate(docid="d3", score=0.0, doc={"text": "How to bake sourdough bread at home."}),
    ],
)
result = reranker.rerank(request, rank_start=0, rank_end=3)
for i, candidate in enumerate(result.candidates, start=1):
    print(f"{i}. {candidate.docid} score={candidate.score:.6f}")
PY
```

First run downloads the MonoT5 model from Hugging Face.

## pip fallback

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install rank-llm
```

## Diagnose environment

```bash
python -m rank_llm.doctor
```

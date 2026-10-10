# MCP candidate fixture

`nfcorpus_cats.json` contains one actual `retrieve_and_rerank` result for
`cats`, including all ten candidates with their original IDs, BM25 scores,
stored document fields, and model-produced order. The standalone `rerank` test
loads its candidates directly so it exercises the same kind of input as the
original chained test without depending on another test's output.

Captured on 2026-10-09 using the unchanged pipeline at commit
`d4a158caf40d68be24c8f370951db6d10b0c2843`, Python 3.12.13, JDK 21,
Pyserini 2.4.0, and vLLM 0.20.1. The model snapshot was
`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`.

The prebuilt index was `beir-v1.0.0-nfcorpus.flat`, archive
`lucene-inverted.beir-v1.0.0-nfcorpus.flat.20221116.505594.tar.gz`, MD5
`eb7a6f1bb15071c2940bc50752d86626`.
[Source index](https://huggingface.co/datasets/castorini/prebuilt-indexes-beir/tree/main/lucene-inverted/flat)

To capture a new snapshot, run this in a synchronous Python entry point with
Pyserini, JDK 21, the model, and a suitable GPU available:

```python
from rank_llm.cli.operations import run_mcp_retrieve_and_rerank
from rank_llm.retrieve import RetrievalMethod

run_mcp_retrieve_and_rerank(
    model_path="Qwen/Qwen3-0.6B",
    query="cats",
    dataset="beir-v1.0.0-nfcorpus.flat",
    output_jsonl_file="nfcorpus_cats.jsonl",
    output_trec_file="nfcorpus_cats.trec",
    retrieval_method=RetrievalMethod.BM25,
    top_k_candidates=10,
    max_queries=1,
)
```

Pretty-print the single JSONL record as `nfcorpus_cats.json`. Model ordering may
vary between captures; the fixture is test input, not a ranking-quality oracle.

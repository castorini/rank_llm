import json
from pathlib import Path

import modal

source = Path(__file__).resolve().parents[1] / "src"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.16.1",
        "accelerate>=1.0",
        "huggingface_hub>=0.34",
        "dacite>=1.8.1",
        "ftfy>=6.1.1",
        "pyyaml>=6.0",
        "tqdm>=4.66",
    )
    .env({"PYTHONPATH": "/opt/rank_llm/src", "HF_HOME": "/cache"})
    .add_local_dir(str(source), "/opt/rank_llm/src")
)
cache = modal.Volume.from_name("gaggle-hf-cache", create_if_missing=True)
app = modal.App("gaggle-rerank", image=image)


@app.function(gpu="A100-40GB", memory=32768, timeout=3600, volumes={"/cache": cache})
def rerank(model: str, query: str, passages: list[str]) -> list[dict]:
    from rank_llm.data import Candidate, Query, Request
    from rank_llm.rerank.reranker import Reranker

    coordinator = Reranker.create_model_coordinator(
        model, None, False, device="cuda", batch_size=16
    )
    cache.commit()
    request = Request(
        Query(query, "q1"),
        [
            Candidate(str(index), 0, {"text": passage})
            for index, passage in enumerate(passages)
        ],
    )
    result = coordinator.rerank_batch([request])[0]
    return [
        {
            "index": int(candidate.docid),
            "passage": candidate.doc["text"],
            "score": candidate.score,
        }
        for candidate in result.candidates
    ]


@app.local_entrypoint()
def main(input_path: str, model: str = "castorini/gaggle-reranker-20261005"):
    data = json.loads(Path(input_path).read_text())
    query = data["query"]
    passages = data["passages"]
    if not isinstance(query, str) or not isinstance(passages, list) or not passages:
        raise ValueError("Input must contain a query and a nonempty list of passages")
    if not all(isinstance(passage, str) for passage in passages):
        raise ValueError("Every passage must be a string")
    print(json.dumps(rerank.remote(model, query, passages), indent=2))

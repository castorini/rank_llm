import json
import subprocess
from pathlib import Path

import modal

root = Path(__file__).resolve().parents[1]
image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("openjdk-21-jdk-headless")
    .pip_install("torch==2.9.1", "transformers==5.16.1", "accelerate>=1.0")
    .add_local_file(
        str(root / "pyproject.toml"), "/opt/rank_llm/pyproject.toml", copy=True
    )
    .add_local_file(
        str(root / "requirements.txt"), "/opt/rank_llm/requirements.txt", copy=True
    )
    .add_local_file(str(root / "README.md"), "/opt/rank_llm/README.md", copy=True)
    .add_local_file(str(root / "LICENSE"), "/opt/rank_llm/LICENSE", copy=True)
    .add_local_dir(str(root / "src"), "/opt/rank_llm/src", copy=True)
    .run_commands("pip install -e '/opt/rank_llm[local,pyserini]' 'torch==2.9.1'")
    .workdir("/opt/rank_llm")
    .env(
        {
            "HF_HOME": "/cache",
            "PYSERINI_CACHE": "/cache/pyserini",
            "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64",
        }
    )
)
cache = modal.Volume.from_name("gaggle-hf-cache", create_if_missing=True)
app = modal.App("gaggle-rerank", image=image)


@app.function(gpu="A100-40GB", memory=32768, timeout=3600, volumes={"/cache": cache})
def rerank(model_path: str, payload: dict, batch_size: int) -> dict:
    command = [
        "rank-llm",
        "--output",
        "json",
        "rerank",
        "--model-path",
        model_path,
        "--input-json",
        json.dumps(payload),
        "--batch-size",
        str(batch_size),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    cache.commit()
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout.strip().splitlines()[-1])


@app.local_entrypoint()
def main(
    input_path: str,
    model_path: str = "castorini/gaggle-reranker-20261005",
    batch_size: int = 16,
):
    payload = json.loads(Path(input_path).read_text())
    print(json.dumps(rerank.remote(model_path, payload, batch_size), indent=2))

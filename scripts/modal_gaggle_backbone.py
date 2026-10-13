import json
from pathlib import Path

import modal

CAUSAL = "castorini/gaggle-nanochat-pretrained-climbmix-20260924"
BIDIRECTIONAL = "castorini/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929"
CHECKPOINTS = {
    CAUSAL: ("363ba278c2ef13f2997a4d97df793b1dd327efda", 227158),
    BIDIRECTIONAL: ("b6ec9e6c41b12425ed0a411bfee1f59c5a18923d", 229067),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        "huggingface_hub>=0.34,<1",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
        "filelock>=3.19.0",
        "kernels>=0.11.7",
    )
    .run_commands(
        "git clone https://github.com/karpathy/nanochat.git /opt/nanochat",
        "cd /opt/nanochat && git checkout 92d63d4",
    )
    .env({"PYTHONPATH": "/opt/nanochat"})
)
cache = modal.Volume.from_name("gaggle-hf-cache", create_if_missing=True)
app = modal.App("gaggle-backbone", image=image)


@app.function(
    gpu="A100-40GB",
    memory=65536,
    timeout=3600,
    volumes={"/cache": cache},
    single_use_containers=True,
)
def score(model_name: str, prompt: str, continuations: list[str]) -> list[dict]:
    import os

    import torch
    from huggingface_hub import snapshot_download

    revision, step = CHECKPOINTS[model_name]
    path = snapshot_download(
        model_name,
        revision=revision,
        local_dir=f"/cache/{model_name.rsplit('/', 1)[-1]}",
    )
    cache.commit()
    os.environ["NANOCHAT_BASE_DIR"] = path
    bidirectional = model_name == BIDIRECTIONAL
    if bidirectional:
        os.environ["NANOCHAT_BIDIRECTIONAL"] = "1"

    import nanochat

    if bidirectional:
        nanochat.__path__.insert(0, os.path.join(path, "code", "nanochat"))
    from nanochat.checkpoint_manager import build_model

    model, tokenizer, _ = build_model(
        path, step=step, device=torch.device("cuda"), phase="eval"
    )
    ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    scores = []
    for continuation in continuations:
        tokens = tokenizer.encode(continuation)
        if bidirectional:
            mask = tokenizer.encode_special("<|output_end|>")
            inputs = ids + [mask] * len(tokens)
            targets = [-1] * (len(ids) - 1) + tokens + [-1]
        else:
            inputs = ids + tokens[:-1]
            targets = [-1] * (len(ids) - 1) + tokens
        x = torch.tensor([inputs], device="cuda")
        y = torch.tensor([targets], device="cuda")
        with torch.inference_mode():
            logprob = -model(x, targets=y, loss_reduction="sum").item()
        scores.append({"continuation": continuation, "logprob": logprob})
    return sorted(scores, key=lambda item: item["logprob"], reverse=True)


@app.local_entrypoint()
def main(input_path: str, model: str = CAUSAL):
    if model not in CHECKPOINTS:
        raise ValueError(f"Unknown backbone: {model}")
    data = json.loads(Path(input_path).read_text())
    prompt = data["prompt"]
    continuations = data["continuations"]
    if (
        not isinstance(prompt, str)
        or not isinstance(continuations, list)
        or not continuations
    ):
        raise ValueError(
            "Input must contain a prompt and a nonempty list of continuations"
        )
    if not all(
        isinstance(continuation, str) and continuation for continuation in continuations
    ):
        raise ValueError("Every continuation must be a nonempty string")
    print(json.dumps(score.remote(model, prompt, continuations), indent=2))

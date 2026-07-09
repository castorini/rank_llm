"""Remote bootstrap executed on a Google Colab runtime via ``colab exec -f``.

This file is sent verbatim to a provisioned Colab kernel by the local
orchestration scripts (``train_on_colab.sh`` / ``rerank_on_colab.sh``).

``colab exec`` only transmits a single file and provides no way to pass argv,
so the orchestration script prepends a line of the form::

    CONFIG = {"mode": "train", "base_model": "...", ...}

before the contents of this file and executes the concatenation. The merge
below lets that injected dict override the defaults while keeping this file
independently importable / lintable (``CONFIG`` is always defined).

The script clones rank_llm, installs the right extras, runs the requested
training or reranking command, and prints ``ARTIFACT_PATH=<remote path>`` on
the last line so the caller knows what to ``colab download``.
"""

import os
import shlex
import subprocess
import sys

DEFAULT_CONFIG = {
    # Common
    "mode": "train",  # "train" | "rerank"
    "repo_url": "https://github.com/castorini/rank_llm.git",
    "ref": "main",
    "workdir": "/content/rank_llm",
    "install_flash_attn": False,
    # Training
    "base_model": "HuggingFaceH4/zephyr-7b-beta",
    "train_dataset_path": "rryisthebest/rank_zephyr_training_data_alpha",
    "objective": "generation",  # generation | ranking | combined
    "output_dir": "/content/rank_llm/training/models/colab_run",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_warmup_steps": 10,
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "extra_train_args": [],  # e.g. ["--ranking_loss", "ranknet", "--weighted"]
    # Reranking
    "model_path": "castorini/rank_zephyr_7b_v1_full",
    "rerank_mode": "dataset",  # "dataset" | "requests_url"
    "dataset": "dl19",
    "retrieval_method": "SPLADE++_EnsembleDistil_ONNX",
    "requests_url": "",  # used when rerank_mode == "requests_url"
    "top_k": 100,
    "context_size": 4096,
    "num_gpus": 1,
    "install_pyserini": True,  # needed for dataset-mode retrieval (JDK 21)
    "extra_rerank_args": [],
}

# The orchestration script may inject a partial ``CONFIG`` before this file;
# merge it over the defaults. Accessing via ``globals()`` keeps this valid when
# the file is run standalone (no injection) so linters/py_compile stay happy.
CONFIG = {**DEFAULT_CONFIG, **globals().get("CONFIG", {})}


def run(cmd, cwd=None, env=None):
    """Run a command, streaming its output; raise on non-zero exit.

    Colab's kernel only surfaces the Python ``sys.stdout``, not a child
    process's OS-level stdout/stderr. So we merge the child's streams and
    re-print them through ``print`` line by line — otherwise subprocess errors
    (which go to stderr) are invisible in the ``colab exec`` output.
    """
    printable = cmd if isinstance(cmd, str) else shlex.join(cmd)
    print(f"\n>>> {printable}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        shell=isinstance(cmd, str),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"Command failed (exit {proc.returncode}): {printable}")


def clone_repo():
    workdir = CONFIG["workdir"]
    if os.path.isdir(os.path.join(workdir, ".git")):
        print(f"Repo already present at {workdir}, skipping clone.", flush=True)
        return
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "-b",
            str(CONFIG["ref"]),
            CONFIG["repo_url"],
            workdir,
        ]
    )


def pip_install(spec, editable=False):
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if editable:
        cmd.append("-e")
    cmd.append(spec)
    run(cmd)


def install_apt_jdk21():
    # rank_llm retrieval (pyserini/anserini) requires JDK 21.
    run("apt-get update -qq && apt-get install -y -qq openjdk-21-jdk-headless")


def archive(path):
    """Tar a result directory into a single file for a clean `colab download`."""
    path = path.rstrip("/")
    tar = path + ".tar.gz"
    parent = os.path.dirname(path) or "."
    base = os.path.basename(path)
    run(["tar", "czf", tar, "-C", parent, base])
    return tar


def do_train():
    clone_repo()
    workdir = CONFIG["workdir"]
    pip_install(f"{workdir}[training]", editable=True)
    if CONFIG["install_flash_attn"]:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "flash-attn",
                "--no-build-isolation",
            ]
        )

    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    env = dict(os.environ)
    env["DS_SKIP_CUDA_CHECK"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        "configs/accel_config_single_gpu.yaml",
        "train_rankllm.py",
        "--model_name_or_path",
        str(CONFIG["base_model"]),
        "--train_dataset_path",
        str(CONFIG["train_dataset_path"]),
        "--objective",
        str(CONFIG["objective"]),
        "--num_train_epochs",
        str(CONFIG["num_train_epochs"]),
        "--per_device_train_batch_size",
        str(CONFIG["per_device_train_batch_size"]),
        "--gradient_accumulation_steps",
        str(CONFIG["gradient_accumulation_steps"]),
        "--num_warmup_steps",
        str(CONFIG["num_warmup_steps"]),
        "--lr_scheduler_type",
        str(CONFIG["lr_scheduler_type"]),
        "--seed",
        str(CONFIG["seed"]),
        "--gradient_checkpointing",
        "--low_cpu_mem_usage",
        "--output_dir",
        output_dir,
        "--checkpointing_steps",
        "epoch",
    ]
    cmd += [str(a) for a in CONFIG["extra_train_args"]]
    run(cmd, cwd=os.path.join(workdir, "training"), env=env)
    return archive(output_dir)


def do_rerank():
    clone_repo()
    workdir = CONFIG["workdir"]
    pyserini = CONFIG["install_pyserini"] and CONFIG["rerank_mode"] == "dataset"
    spec = f"{workdir}[vllm,pyserini]" if pyserini else f"{workdir}[vllm]"
    if pyserini:
        install_apt_jdk21()
    pip_install(spec, editable=True)
    # ONNX learned-sparse retrievers (e.g. SPLADE++_EnsembleDistil_ONNX) need
    # onnxruntime for the query encoder; the pyserini extra does not pull it in.
    if pyserini and "onnx" in str(CONFIG["retrieval_method"]).lower():
        pip_install("onnxruntime")

    out_dir = os.path.join(workdir, "colab_rerank_out")
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "results.jsonl")
    out_trec = os.path.join(out_dir, "results.trec")

    cmd = [
        "rank-llm",
        "rerank",
        "--model-path",
        str(CONFIG["model_path"]),
        "--num-gpus",
        str(CONFIG["num_gpus"]),
        "--context-size",
        str(CONFIG["context_size"]),
        "--output-jsonl-file",
        out_jsonl,
        "--output-trec-file",
        out_trec,
    ]

    if CONFIG["rerank_mode"] == "requests_url":
        req_file = os.path.join(out_dir, "requests.jsonl")
        run(["wget", "-q", "-O", req_file, str(CONFIG["requests_url"])])
        cmd += ["--requests-file", req_file]
    else:
        cmd += [
            "--dataset",
            str(CONFIG["dataset"]),
            "--retrieval-method",
            str(CONFIG["retrieval_method"]),
            "--top-k-candidates",
            str(CONFIG["top_k"]),
        ]

    cmd += [str(a) for a in CONFIG["extra_rerank_args"]]
    # Unbuffered so the child's stdout (model/retrieval progress) is interleaved
    # with stderr in real time instead of being lost if the process crashes.
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # pyserini 1.2.0 eagerly builds an ``openai.OpenAI()`` client at import time
    # (pyserini/encode/_openai.py). With openai>=2 that raises when no key is set
    # — even for bm25 retrieval that never touches OpenAI. A placeholder lets the
    # import succeed; the client is never actually called. A real key is kept.
    env.setdefault("OPENAI_API_KEY", "sk-pyserini-import-placeholder")
    run(cmd, cwd=workdir, env=env)
    return archive(out_dir)


def main():
    mode = CONFIG["mode"]
    if mode == "train":
        artifact = do_train()
    elif mode == "rerank":
        artifact = do_rerank()
    else:
        raise SystemExit(f"Unknown mode: {mode!r} (expected 'train' or 'rerank')")
    # Last line is parsed by the orchestration script for `colab download`.
    print(f"\nARTIFACT_PATH={artifact}", flush=True)


if __name__ == "__main__":
    main()

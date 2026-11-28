"""Environment setup for running the RankLLM onboarding guides on a Modal A100.

This is the Modal counterpart of scripts/colab_setup.py: it builds the environment, and you
run the guide commands yourself in a shell. From the repo root:

    modal run scripts/modal_setup.py            # setup check on CPU, ends with "setup ok"
    modal shell scripts/modal_setup.py::shell   # A100 shell for the lessons, cd /opt/rank_llm
    modal volume get rank-llm-onboarding-cache /rerank_results .

The image copies this checkout's src/, so what you run reflects your local HEAD.
Image and volume pattern forked from scripts/modal_gaggle.py (#463).
"""

import subprocess
from importlib.metadata import version
from pathlib import Path

import modal

root = Path(__file__).resolve().parents[1]
CU128 = "https://download.pytorch.org/whl/cu128"
PACKAGES = ("rank-llm", "vllm", "torch", "transformers", "pyserini")

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("openjdk-21-jdk-headless")
    .add_local_file(str(root / "pyproject.toml"), "/opt/rank_llm/pyproject.toml", copy=True)
    .add_local_file(str(root / "requirements.txt"), "/opt/rank_llm/requirements.txt", copy=True)
    .add_local_file(str(root / "README.md"), "/opt/rank_llm/README.md", copy=True)
    .add_local_file(str(root / "LICENSE"), "/opt/rank_llm/LICENSE", copy=True)
    .add_local_dir(str(root / "src"), "/opt/rank_llm/src", copy=True)
    # The install and pin steps from the Colab section of docs/onboarding-rz.md, in the same
    # order. The pins are kept so results compare with the guides' reproduction tables.
    .run_commands("pip install -e '/opt/rank_llm[vllm,pyserini]' onnxruntime")
    .run_commands(
        f"pip install --force-reinstall 'vllm==0.11.0' 'torch==2.8.0' --extra-index-url {CU128}"
    )
    .run_commands("pip install 'transformers==4.55.4'")
    .workdir("/opt/rank_llm")
    .env(
        {
            "HF_HOME": "/cache/hf",
            "PYSERINI_CACHE": "/cache/pyserini",
            "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64",
            # vLLM forks its engine process after CUDA is initialized; spawn avoids
            # "Cannot re-initialize CUDA in forked subprocess".
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            # The FlashInfer sampler JIT-compiles with nvcc, which the image lacks.
            # vLLM 0.11.0 reads this with int(), so it must be "0". The guide commands
            # decode at temperature 0.0, where vLLM takes the argmax before this sampler.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    # Triton compiles a C launcher stub at runtime; ubuntu:24.04 has no compiler.
    .apt_install("build-essential")
)
cache = modal.Volume.from_name("rank-llm-onboarding-cache", create_if_missing=True)
app = modal.App("rank-llm-onboarding", image=image)


@app.function(timeout=900, scaledown_window=2, volumes={"/cache": cache})
def smoke():
    """CPU only: checks the image, pins, Java, Pyserini and the CLI before any GPU time."""
    import torch
    from pyserini.search.lucene import LuceneImpactSearcher

    for package in PACKAGES:
        print(f"{package:14s} {version(package)}")
    print("torch cuda build:", torch.version.cuda)
    java = subprocess.run(["java", "-version"], capture_output=True, text=True)
    print(java.stderr.strip().splitlines()[0])
    print("pyserini imports OK:", LuceneImpactSearcher is not None)
    cli = subprocess.run(["rank-llm", "--help"], capture_output=True, text=True)
    print("rank-llm --help exit code:", cli.returncode)
    if cli.returncode != 0:
        raise RuntimeError(cli.stderr or cli.stdout)
    print("\nsetup ok")


@app.local_entrypoint()
def main():
    smoke.remote()


@app.function(
    gpu="A100-40GB",
    memory=32768,
    # modal shell caps interactive sessions at 1 hour regardless of this value
    # (modal 1.6.3, cli/shell.py).
    timeout=3600,
    volumes={"/cache": cache},
)
def shell():
    """Target for `modal shell scripts/modal_setup.py::shell`; runs nothing itself.

    The shell gets this function's image, GPU, memory and volume. Run the guide commands from
    /opt/rank_llm, copy rerank_results/ to /cache/ before you exit, and fetch it from your
    machine with `modal volume get`.
    """
    pass

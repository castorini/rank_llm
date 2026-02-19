import subprocess


def find_best_gpu(utilization_fraction: float) -> str:
    """
    Find GPU with enough free memory for the requested utilization.

    Usage:
        os.environ["CUDA_VISIBLE_DEVICES"] = find_best_gpu(0.8)
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        print("Warning: nvidia-smi failed. Defaulting to GPU 0.")
        return "0"

    for line in output.strip().split("\n"):
        try:
            idx, total, free = line.split(", ")
            if float(total) * utilization_fraction <= float(free):
                print(f"Selected GPU {idx} (Free: {float(free)/1024:.1f}GB)")
                return idx
        except ValueError:
            continue

    raise RuntimeError(f"No GPU found fitting utilization {utilization_fraction}.")

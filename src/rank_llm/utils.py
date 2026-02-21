import subprocess


def find_best_gpus(utilization_fraction: float) -> list[str]:
    """
    Find GPUs with enough free memory for the requested utilization,
    ordered from least to most utilized.

    Usage:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(find_best_gpus(0.8))
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
        return ["0"]

    candidates = []
    for line in output.strip().split("\n"):
        try:
            idx, total, free = line.split(", ")
            if float(total) * utilization_fraction <= float(free):
                candidates.append((idx, float(free)))
        except ValueError:
            continue

    if not candidates:
        raise RuntimeError(f"No GPU found fitting utilization {utilization_fraction}.")

    candidates.sort(key=lambda x: x[1], reverse=True)
    gpus = [idx for idx, _ in candidates]
    print(f"Selected GPUs {gpus} (ordered least to most utilized)")
    return gpus

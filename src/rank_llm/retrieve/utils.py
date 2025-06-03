import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

logger = logging.getLogger(__name__)


# https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_cache_home():
    custom_dir = os.environ.get("RANK_LLM_CACHE")
    if custom_dir is not None and custom_dir != "":
        print("custom")
        Path(custom_dir).mkdir(exist_ok=True)
        return custom_dir

    default_dir = "."
    Path(default_dir).mkdir(exist_ok=True)
    return default_dir


def download_cached_hits(
    query_name: str,
) -> str:
    """
    Download stored retrieved_results from HuggingFace datasets repo.

    Args:
        query_name: query name (eg. "BM25/retrieve_results_arguana_top100.jsonl")

    Returns:
        Local path to the downloaded file
    """
    repo_id = "castorini/rank_llm_data"
    hf_filename = f"retrieve_results/{query_name}"
    cache_dir = get_cache_home()

    file_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=hf_filename,
        local_dir=cache_dir,
    )

    return file_path

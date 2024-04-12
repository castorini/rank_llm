import json
import os
import urllib.request
from urllib.parse import urlparse

from pyserini.util import *


def no_bool_convert(pairs):
    return {k: str(v).casefold() if isinstance(v, bool) else v for k, v in pairs}


def parse_file_info(file_name, file_url, save_dir):
    # download file
    local_file_path = download_url(
        file_url, save_dir, local_filename=file_name, verbose=True
    )

    # Compute MD5
    md5 = compute_md5(local_file_path)

    # Get file size
    size = os.path.getsize(local_file_path)

    # Remove the downloaded file
    os.remove(local_file_path)

    # Placeholder description
    description = "Sample description"

    return description, md5, size


# Save directory
def get_repo_info(repo_url, save_dir):
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL")

    owner, repo = path_parts[:2]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    with urllib.request.urlopen(api_url) as response:
        data = json.loads(response.read().decode())

    hits_info = {}
    for file in data.get("tree", []):
        if file["type"] == "blob":
            file_name = file["path"]
            file_url = f"https://github.com/{owner}/{repo}/raw/main/{file_name}"
            description, md5, size = parse_file_info(file_name, file_url, save_dir)
            hits_info["/".join(file_name.rsplit("/", 2)[-2:])] = {
                "description": description,
                "urls": [file_url],
                "md5": md5,
                "size (bytes)": size,
                "downloaded": False,
            }
    return hits_info


repo_url = "https://github.com/castorini/rank_llm_data/tree/main/retrieve_results"
save_dir = "retrieve_results/SPLADE_P_P_ENSEMBLE_DISTIL"
info = get_repo_info(repo_url, save_dir)
info_json = json.dumps(info, indent=4)

with open("repo_info.txt", "w") as f:
    f.write(f"HITS_INFO = {info_json}\n")

with open("repo_info.txt", "r") as f:
    data = f.read()
    data = data.replace("false", "False")

with open("repo_info.py", "w") as f:
    f.write(data)

os.remove("repo_info.txt")

import hashlib
import logging
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from tqdm import tqdm

from rank_llm.retrieve.repo_info import HITS_INFO

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


# For large files, we need to compute MD5 block by block. See:
# https://stackoverflow.com/questions/1131220/get-md5-hash-of-big-files-in-python
def compute_md5(file, block_size=2**20):
    m = hashlib.md5()
    with open(file, "rb") as f:
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()


def download_url(
    url, save_dir, local_filename=None, md5=None, force=False, verbose=True
):
    # If caller does not specify local filename, figure it out from the download URL:
    if not local_filename:
        filename = url.split("/")[-1]
        filename = re.sub(
            "\\?dl=1$", "", filename
        )  # Remove the Dropbox 'force download' parameter
    else:
        # Otherwise, use the specified local_filename:
        filename = local_filename

    destination_path = os.path.join(save_dir, filename)

    if verbose:
        print(f"curr_path{os.getcwd()}")
        print(f"Downloading {url} to {destination_path}...")

    # Check to see if file already exists, if so, simply return (quietly) unless force=True, in which case we remove
    # destination file and download fresh copy.
    if os.path.exists(destination_path):
        if verbose:
            print(f"{destination_path} already exists!")
        if not force:
            if verbose:
                print(f"Skipping download.")
            return destination_path
        if verbose:
            print(f"force=True, removing {destination_path}; fetching fresh copy...")
        os.remove(destination_path)

    with TqdmUpTo(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename
    ) as t:
        urlretrieve(url, filename=destination_path, reporthook=t.update_to)

    if md5:
        md5_computed = compute_md5(destination_path)
        assert (
            md5_computed == md5
        ), f"{destination_path} does not match checksum! Expecting {md5} got {md5_computed}."

    return destination_path


def get_cache_home():
    custom_dir = os.environ.get("RANK_LLM_CACHE")
    if custom_dir is not None and custom_dir != "":
        print("custom")
        return custom_dir
    return os.path.expanduser(
        os.path.join(f"~{os.path.sep}rank_llm", "retrieve_results")
    )


def download_and_unpack_hits(
    url,
    hits_directory="files",
    local_filename=False,
    force=False,
    verbose=True,
    prebuilt=False,
    md5=None,
):
    # If caller does not specify local filename, figure it out from the download URL:
    if not local_filename:
        file_name = url.split("/")[-1]
    else:
        # Otherwise, use the specified local_filename:
        file_name = local_filename

    if prebuilt:
        hits_directory = os.path.join(get_cache_home(), hits_directory)
        file_path = os.path.join(hits_directory, f"{file_name}.{md5}")

        if not os.path.exists(hits_directory):
            os.makedirs(hits_directory)

        local_file = os.path.join(hits_directory, file_name)
    else:
        local_file = os.path.join(hits_directory, file_name)
        file_path = os.path.join(hits_directory, file_name)

    # Check to see if file already exists, if so, simply return (quietly) unless force=True, in which case we remove
    # file and download fresh copy.
    if os.path.exists(file_path):
        if not force:
            if verbose:
                print(f"{file_path} already exists, skipping download.")
            return file_path
        if verbose:
            print(
                f"{file_path} already exists, but force=True, removing {file_path} and fetching fresh copy..."
            )
        os.remove(file_path)

    print(f"Downloading file at {url}...")
    download_url(
        url, hits_directory, local_filename=local_filename, verbose=False, md5=md5
    )

    # No need to extract for JSON and text files
    if verbose:
        print(f"File {local_file} has been downloaded to {file_path}.")
    return local_file


def download_cached_hits(query_name, force=False, verbose=True, mirror=None):
    if query_name not in HITS_INFO:
        print(f"query_name unrecognized {query_name}")
        raise ValueError(f"Unrecognized query name {query_name}")
    query_md5 = HITS_INFO[query_name]["md5"]
    for url in HITS_INFO[query_name]["urls"]:
        try:
            hits_dir = query_name.rsplit("/", 2)[-2]
            return download_and_unpack_hits(
                url, hits_directory=hits_dir, prebuilt=True, md5=query_md5
            )
        except (HTTPError, URLError) as e:
            print(
                f"Unable to download cached retrieved hits at {url}, trying next URL..."
            )
    raise ValueError(f"Unable to download cached retrieved hits at any known URLs.")


get_cache_home()

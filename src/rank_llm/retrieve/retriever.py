import glob
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dacite import from_dict

from rank_llm.data import Request
from rank_llm.retrieve.utils import download_cached_hits, get_cache_home

from . import PyseriniRetriever, RetrievalMethod


class RetrievalMode(Enum):
    DATASET = "dataset"
    CUSTOM = "custom"
    LOAD_FROM_FILE = "load_from_file"

    def __str__(self):
        return self.value


class Retriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        retrieval_method: RetrievalMethod = RetrievalMethod.UNSPECIFIED,
        query: str = None,
        index_path: str = None,
        topics_path: str = None,
        index_type: str = None,
        encoder: str = None,
        onnx: bool = False,
    ) -> None:
        self._retrieval_mode = retrieval_mode
        self._dataset = dataset
        self._retrieval_method = retrieval_method
        self._query = query
        self._index_path = index_path
        self._topics_path = topics_path
        self._index_type = index_type
        self._encoder = encoder
        self._onnx = onnx

    @staticmethod
    def from_dataset_with_prebuilt_index(
        dataset_name: str,
        retrieval_method: RetrievalMethod = RetrievalMethod.BM25,
        k: int = 100,
    ):
        """
        Creates a Retriever instance for a dataset with a prebuilt index.

        Args:
            dataset_name (str): The name of the dataset.
            retrieval_method (RetrievalMethod): The retrieval method to be used. Defaults to BM25.
            k (int, optional): The top k hits to retrieve. Defaults to 100.

        Returns:
            List[Request]: The list of requests. Each request has a query and list of candidates

        Raises:
            ValueError: If dataset name or retrieval method is invalid or missing.
        """
        if not dataset_name:
            raise ValueError("Please provide name of the dataset.")
        if retrieval_method == RetrievalMethod.UNSPECIFIED:
            raise ValueError(
                f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method."
            )
        retriever = Retriever(
            RetrievalMode.DATASET,
            dataset=dataset_name,
            retrieval_method=retrieval_method,
        )
        return retriever.retrieve(k=k)

    @staticmethod
    def from_custom_index(
        index_path: str,
        topics_path: str,
        index_type: str,
        encoder: str = None,
        onnx: bool = False,
        k: int = 100,
    ):
        """
        Creates a Retriever instance for a dataset with a prebuilt index.

        Args:
            index_path (str): The path to the lucene or impact index.
            topics_path (str): The path to the topics file.
            index_type (str): Index type; choices: [lucene, impact].
            encoder (str, optional): The encoder used in impact indexes. Defaults to None.
            onnx (bool, optional): Flag for using onnx in impact indexes. Defaults to False.
            k (int, optional): The top k hits to retrieve. Defaults to 100.

        Returns:
            List[Request]: The list of requests. Each request has a query and list of candidates

        Raises:
            ValueError: If index_path or topics_path are invalid paths and if index_type is not lucene or impact
        """
        if not index_path:
            raise ValueError("Please provide a path to the index")
        if not topics_path:
            raise ValueError("Please provide a path to the topics file")
        if index_type not in ["lucene", "impact"]:
            raise ValueError(f"index_type must be [lucene, impact], not {index_type}")

        # implied from name of index and topic dir
        index_name = os.path.basename(os.path.normpath(index_path))
        topics_name = os.path.basename(os.path.normpath(topics_path))
        dataset_name = f"index-{index_name}_topic-{topics_name}_type-{index_type}_encoder-{encoder}_onnx-{onnx}"
        retriever = Retriever(
            retrieval_mode=RetrievalMode.CUSTOM,
            dataset=dataset_name,
            retrieval_method=RetrievalMethod.CUSTOM_INDEX,
            index_path=index_path,
            topics_path=topics_path,
            index_type=index_type,
            encoder=encoder,
            onnx=onnx,
        )
        return retriever.retrieve(k=k)

    def _get_file_with_highest_k(
        self,
        retrieve_results_dirname: str,
        retrieval_method_name: str,
        dataset_name: str,
        file_pattern: str = "retrieve_results_{dataset_name}_top{k}.jsonl",
    ) -> Tuple[Optional[str], int]:
        """
        Finds the file with the highest `k` value in a directory.

        Args:
            retrieve_results_dirname: Base directory (e.g., "retrieve_results").
            retrieval_method_name: Method name (e.g., "BM25").
            dataset_name: Dataset name (e.g., "dl19").
            file_pattern: Format string for filenames (default matches the original pattern).

        Returns:
            Path to the file with the highest `k`, or `None` if no files exist.
        """
        glob_pattern = os.path.join(
            retrieve_results_dirname,
            "retrieve_results",
            retrieval_method_name,
            file_pattern.format(dataset_name=dataset_name, k="*"),
        )

        matching_files = glob.glob(glob_pattern)

        if not matching_files:
            return None, -1

        def _extract_k(file_path):
            match = re.search(rf"top(\d+)\.jsonl$", file_path)
            return int(match.group(1))

        file_with_max_k = max(matching_files, key=_extract_k)
        return file_with_max_k, _extract_k(file_with_max_k)

    def retrieve(self, k: int = 100) -> List[Request]:
        """
        Executes the retrieval process based on the configation provided with the Retriever instance.

        Returns:
            List[Request]: The list of requests. Each request has a query and list of candidates

        Raises:
            ValueError: If the retrieval mode is invalid or the result format is not as expected.
        """
        retrieve_results_dirname = get_cache_home()

        if self._retrieval_mode == RetrievalMode.DATASET:
            max_k_file, max_k = self._get_file_with_highest_k(
                retrieve_results_dirname, self._retrieval_method.name, self._dataset
            )
            if (
                max_k_file and max_k >= k
            ):  # try to see if retrieving from local file works
                try:
                    with open(max_k_file, "r") as f:
                        results = [
                            from_dict(data_class=Request, data=json.loads(line))
                            for line in f
                        ]
                        for result in results:
                            result.candidates = result.candidates[:k]
                        print(
                            f"Successfully loaded {max_k_file} to get top {k} candidates"
                        )
                        return results
                except Exception as local_error:
                    print(f"Local file invalid, attempting HF download: {local_error}")
            try:  # fallback #1: download from HF repo
                query_name = f"{self._retrieval_method.name}/retrieve_results_{self._dataset}_top{100 if k <= 100 else 1000}.jsonl"
                cached_file = download_cached_hits(query_name)

                with open(cached_file, "r") as f:
                    results = [
                        from_dict(data_class=Request, data=json.loads(line))
                        for line in enumerate(f)
                    ]
                    for result in results:
                        result.candidates = result.candidates[:k]
                    print(
                        f"Successfully downloaded cached results to {retrieve_results_dirname}/{cached_file}"
                    )
                    return results
            except Exception as hf_error:
                print(f"HF download failed, using Pyserini: {hf_error}")
            try:  # fallback #2: retrieve on the spot with pyserini
                print(f"Using Pyserini to retrieve with dataset {self._dataset}")
                pyserini = PyseriniRetriever(self._dataset, self._retrieval_method)
                return pyserini.retrieve_and_store(k=k)
            except Exception as pyserini_error:
                raise ValueError(f"Pyserini error: {pyserini_error}")
        elif self._retrieval_mode == RetrievalMode.CUSTOM:
            candidates_file = Path(
                f"{retrieve_results_dirname}/{self._retrieval_method.name}/retrieve_results_{self._dataset}_top{k}.json"
            )
            if not candidates_file.is_file():
                print(f"Retrieving with dataset {self._dataset}")
                pyserini = PyseriniRetriever(
                    dataset=self._dataset,
                    retrieval_method=self._retrieval_method,
                    index_path=self._index_path,
                    topics_path=self._topics_path,
                    index_type=self._index_type,
                    encoder=self._encoder,
                    onnx=self._onnx,
                )
                retrieved_results = pyserini.retrieve_and_store(k=k)
            else:
                print("Reusing existing retrieved results.")
                with open(candidates_file, "r") as f:
                    loaded_results = json.load(f)
                retrieved_results = [
                    from_dict(data_class=Request, data=r) for r in loaded_results
                ]
        else:
            raise ValueError(f"Invalid retrieval mode: {self._retrieval_mode}")
        return retrieved_results

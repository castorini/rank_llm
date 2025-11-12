import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import hf_hub_download

from rank_llm.data import DataWriter, Query, Request, read_requests_from_file
from rank_llm.rerank import IdentityReranker, RankLLM, Reranker
from rank_llm.rerank.reranker import extract_kwargs
from rank_llm.retrieve import (
    TOPICS,
    RetrievalMethod,
    RetrievalMode,
    Retriever,
    ServiceRetriever,
)


def retrieve_and_rerank(
    model_path: str,
    query: str,
    dataset: Union[str, List[str], List[Dict[str, Any]]],
    retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
    retrieval_method: RetrievalMethod = RetrievalMethod.BM25,
    top_k_retrieve: int = 50,
    top_k_rerank: int = 10,
    max_queries: Optional[int] = None,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    qid: int = 1,
    num_passes: int = 1,
    interactive: bool = False,
    default_model_coordinator: RankLLM = None,
    **kwargs: Any,
):
    """Retrieve candidates using Anserini API and rerank them

    Returns:
        - List of top_k_rerank candidates
    """

    # Get reranking model_coordinator
    reranker = Reranker(
        Reranker.create_model_coordinator(
            model_path,
            default_model_coordinator,
            interactive,
            **kwargs,
        )
    )

    # Retrieve initial candidates
    print(f"Retrieving top {top_k_retrieve} passages...")
    requests = retrieve(
        top_k_retrieve,
        interactive,
        retrieval_mode,
        retrieval_method,
        query,
        qid,
        dataset=dataset,
        **kwargs,
    )

    if max_queries is not None:
        requests = requests[: min(len(requests), max_queries)]

    for request in requests:
        request.candidates = request.candidates[:top_k_retrieve]

    # Reranking stages
    print(f"Reranking and returning {top_k_rerank} passages with {model_path}...")
    if reranker.get_model_coordinator() is None:
        # No reranker. IdentityReranker leaves retrieve candidate results as is or randomizes the order.
        shuffle_candidates = True if model_path == "rank_random" else False
        rerank_results = IdentityReranker().rerank_batch(
            requests,
            rank_end=top_k_retrieve,
            shuffle_candidates=shuffle_candidates,
        )
    else:
        # Reranker is of type RankLLM
        for pass_ct in range(num_passes):
            print(f"Pass {pass_ct + 1} of {num_passes}:")
            rerank_results = reranker.rerank_batch(
                requests,
                rank_end=top_k_retrieve,
                rank_start=0,
                shuffle_candidates=shuffle_candidates,
                logging=print_prompts_responses,
                top_k_retrieve=top_k_retrieve,
                **kwargs,
            )

            if num_passes > 1:
                requests = [
                    Request(copy.deepcopy(r.query), copy.deepcopy(r.candidates))
                    for r in rerank_results
                ]

    for rr in rerank_results:
        rr.candidates = rr.candidates[:top_k_rerank]

    # generate trec_eval file & evaluate for named datasets only
    if isinstance(dataset, str) and reranker.get_model_coordinator() is not None:
        file_name = reranker.write_rerank_results(
            retrieval_method.name,
            rerank_results,
            shuffle_candidates,
            top_k_candidates=top_k_retrieve,
            pass_ct=None if num_passes == 1 else pass_ct,
            window_size=kwargs.get("window_size", None),
            dataset_name=dataset,
            sglang_batched=kwargs.get("sglang_batched", False),
            tensorrt_batched=kwargs.get("tensorrt_batched", False),
        )
        if (
            dataset in TOPICS
            and dataset not in ["news"]
            and TOPICS[dataset] not in ["news"]
        ):
            from rank_llm.evaluation.trec_eval import EvalFunction

            print("Evaluating:")
            EvalFunction.eval(["-c", "-m", "ndcg_cut.1", TOPICS[dataset], file_name])
            EvalFunction.eval(["-c", "-m", "ndcg_cut.5", TOPICS[dataset], file_name])
            EvalFunction.eval(["-c", "-m", "ndcg_cut.10", TOPICS[dataset], file_name])
        else:
            print(f"Skipping evaluation as {dataset} is not in TOPICS.")
    elif (
        retrieval_mode == RetrievalMode.CACHED_FILE
        and reranker.get_model_coordinator() is not None
    ):
        writer = DataWriter(rerank_results)
        keys_and_defaults = [
            ("output_jsonl_file", ""),
            ("output_trec_file", ""),
            ("invocations_history_file", ""),
        ]
        [
            output_jsonl_file,
            output_trec_file,
            invocations_history_file,
        ] = extract_kwargs(keys_and_defaults, **kwargs)
        if output_jsonl_file:
            path = Path(output_jsonl_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer.write_in_jsonl_format(output_jsonl_file)
        if output_trec_file:
            path = Path(output_trec_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer.write_in_trec_eval_format(output_trec_file)
        keys_and_defaults = [("populate_invocations_history", False)]
        [populate_invocations_history] = extract_kwargs(keys_and_defaults, **kwargs)
        if populate_invocations_history:
            if invocations_history_file:
                path = Path(invocations_history_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                writer.write_inference_invocations_history(invocations_history_file)
            else:
                raise ValueError(
                    "--invocations_history_file must be a valid jsonl file to store invocations history."
                )
        keys_and_defaults = [("qrels_file", "")]
        [qrels_file] = extract_kwargs(keys_and_defaults, **kwargs)
        if qrels_file:
            from rank_llm.evaluation.trec_eval import EvalFunction

            print("Evaluating:")
            EvalFunction.from_results(
                rerank_results, qrels_file, ["-c", "-m", "ndcg_cut.1"]
            )
            EvalFunction.from_results(
                rerank_results, qrels_file, ["-c", "-m", "ndcg_cut.5"]
            )
            EvalFunction.from_results(
                rerank_results, qrels_file, ["-c", "-m", "ndcg_cut.10"]
            )

    if interactive:
        return (rerank_results, reranker.get_model_coordinator())
    else:
        return rerank_results


def retrieve(
    top_k_retrieve: int = 50,
    interactive: bool = False,
    retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
    retrieval_method: RetrievalMethod = RetrievalMethod.BM25,
    query: str = "",
    qid: int = 1,
    **kwargs,
):
    """Retrieve initial candidates

    Keyword arguments:
    dataset -- dataset to search if interactive
    top_k_retrieve -- top k candidates to retrieve
    retrieval_mode -- Mode of retrieval
    retrieval_method -- Method of retrieval
    query -- query to retrieve against
    qid - qid of query

    Return: requests -- List[Requests]
    """

    # Retrieve
    if interactive and retrieval_mode != RetrievalMode.DATASET:
        raise ValueError(
            f"Unsupport retrieval mode for interactive retrieval. Currently only DATASET mode is supported."
        )

    requests: List[Request] = []
    if retrieval_mode == RetrievalMode.DATASET:
        dataset: Union[str, List[str], List[Dict[str, Any]]] = kwargs.get(
            "dataset", None
        )
        if dataset == None:
            raise ValueError("Must provide a dataset")

        if interactive:
            host: str = kwargs.get("host", "http://localhost:8081")
            service_retriever = ServiceRetriever(
                retrieval_method=retrieval_method, retrieval_mode=retrieval_mode
            )
            if isinstance(dataset, str):
                dataset = [dataset]
            if isinstance(dataset, list):
                # This check remains the same as the contents check still needs manual iteration
                if all(isinstance(item, dict) for item in dataset):
                    raise ValueError(
                        "List[Dict[str, Any]] dataset input is not supported for interactive retrieval"
                    )

            requests = []
            for ds in dataset:
                # Calls Anserini API
                requests.append(
                    service_retriever.retrieve(
                        dataset=ds,
                        request=Request(query=Query(text=query, qid=qid)),
                        k=top_k_retrieve,
                        host=host,
                    )
                )
        else:
            requests = Retriever.from_dataset_with_prebuilt_index(
                dataset_name=dataset,
                retrieval_method=retrieval_method,
                k=top_k_retrieve,
            )
    elif retrieval_mode == RetrievalMode.CUSTOM:
        keys_and_defaults = [
            ("index_path", None),
            ("topics_path", None),
            ("index_type", None),
        ]
        [index_path, topics_path, index_type] = extract_kwargs(
            keys_and_defaults, **kwargs
        )
        requests = Retriever.from_custom_index(
            index_path=index_path, topics_path=topics_path, index_type=index_type
        )
    elif retrieval_mode == RetrievalMode.CACHED_FILE:
        keys_and_defaults = [
            ("requests_file", ""),
        ]
        [requests_file] = extract_kwargs(keys_and_defaults, **kwargs)
        if not os.path.exists(requests_file):
            print(
                f"Requests file {requests_file} does not exist locally, proceeding to download from huggingface."
            )

            path_parts = requests_file.split("/")
            if len(path_parts) != 3:
                raise ValueError(
                    "Invalid requests_file path for huggingface download, need to be in the format of 'retrieve_results/MODEL_NAME/request_file_name.jsonl"
                )
            model_name = path_parts[1]
            local_dir = os.path.join("retrieve_results", model_name)
            os.makedirs(local_dir, exist_ok=True)

            try:
                local_file_path = hf_hub_download(
                    repo_id="castorini/rank_llm_data",
                    filename=requests_file,
                    repo_type="dataset",
                    local_dir=local_dir,
                )
                print(f"Successfully downloaded requests file to {local_file_path}")
                requests = read_requests_from_file(local_file_path)
            except Exception as e:
                if os.path.exists(local_dir) and not os.listdir(local_dir):
                    os.rmdir(local_dir)
                raise ValueError(
                    f"Error downloading requests file from huggingface: {e}"
                )
        else:
            requests = read_requests_from_file(requests_file)

    return requests

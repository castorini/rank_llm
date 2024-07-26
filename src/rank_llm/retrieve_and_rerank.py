from typing import Any, Dict, List, Union

from rank_llm.data import Query, Request
from rank_llm.rerank import IdentityReranker, RankLLM
from rank_llm.rerank.listwise import (
    PromptMode,
    RankListwiseOSLLM,
    SafeOpenai,
    get_azure_openai_args,
    get_openai_api_key,
)
from rank_llm.retrieve import (
    RetrievalMethod,
    RetrievalMode,
    Retriever,
    ServiceRetriever,
)


def retrieve_and_rerank(
    model_path: str,
    dataset: Union[str, List[str], List[Dict[str, Any]]],
    retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
    retrieval_method: RetrievalMethod = RetrievalMethod.BM25,
    top_k_retrieve: int = 50,
    top_k_rerank: int = 10,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    query: str = "",
    qid: int = 1,
    num_passes: int = 1,
    interactive: bool = False,
    default_agent: RankLLM = None,
    **kwargs: Any,
):
    """Retrieve candidates using Anserini API and rerank them

    Returns:
        - List of top_k_rerank candidates
    """

    # Get reranking agent
    reranker = get_reranker(model_path.lower(), default_agent, interactive, **kwargs)

    # Retrieve initial candidates
    print(f"Retrieving top {top_k_retrieve} passages...")
    requests = retrieve(
        dataset,
        top_k_retrieve,
        interactive,
        retrieval_mode,
        retrieval_method,
        query,
        qid,
        **kwargs,
    )
    print(f"Retrieval complete!")

    # Reranking stage
    print(f"Reranking and returning {top_k_rerank} passages with {model_path}...")
    if reranker is None:
        # No reranker. IdentityReranker leaves retrieve candidate results as is or randomizes the order.
        shuffle_candidates = True if model_path == "rank_random" else False
        rerank_results = IdentityReranker().rerank_batch(
            requests,
            rank_end=top_k_retrieve,
            shuffle_candidates=(shuffle_candidates),
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
    print(f"Reranking with {num_passes} passes complete!")

    for rr in rerank_results:
        rr.candidates = rr.candidates[:top_k_rerank]
    return (rerank_results, reranker)


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
        host: str = kwargs.get("host", "http://localhost:8081")
        dataset: Union[str, List[str], List[Dict[str, Any]]] = kwargs.get(
            "dataset", None
        )
        if dataset == None:
            raise ValueError("Must provide a dataset")

        if interactive:
            service_retriever = ServiceRetriever(
                retrieval_method=retrieval_method, retrieval_mode=retrieval_mode
            )

            # Calls Anserini API
            requests = [
                service_retriever.retrieve(
                    dataset=dataset,
                    request=Request(query=Query(text=query, qid=qid)),
                    k=top_k_retrieve,
                    host=host,
                )
            ]
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

    return requests


def get_reranker(
    model_path: str,
    default_agent: RankLLM,
    interactive: bool,
    **kwargs: Any,
):
    """Construct rerank agent

    Keyword arguments:
    argument -- description
    model_path -- name of model
    default_agent -- used for interactive mode to pass in a pre-instantiated agent to use
    interactive -- whether to run retrieve_and_rerank in interactive mode, used by the API

    Return: rerank agent -- Option<RankLLM>
    """
    use_azure_openai: bool = kwargs.get("use_azure_openai", False)

    if interactive and default_agent is not None:
        # Default rerank agent
        agent = default_agent
    elif "gpt" in model_path or use_azure_openai:
        # GPT based reranking models

        keys_and_defaults = [
            ("context_size", 4096),
            ("prompt_mode", PromptMode.RANK_GPT),
            ("num_few_shot_examples", 0),
            ("window_size", 20),
        ]
        [
            context_size,
            prompt_mode,
            num_few_shot_examples,
            window_size,
        ] = extract_kwargs(keys_and_defaults, **kwargs)

        openai_keys = get_openai_api_key()
        agent = SafeOpenai(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            window_size=window_size,
            num_few_shot_examples=num_few_shot_examples,
            keys=openai_keys,
            **(get_azure_openai_args() if use_azure_openai else {}),
        )
    elif "vicuna" in model_path or "zephyr" in model_path:
        # RankVicuna or RankZephyr model suite
        print(f"Loading {model_path} ...")

        model_full_paths = {
            "rank_zephyr": "castorini/rank_zephyr_7b_v1_full",
            "rank_vicuna": "castorini/rank_vicuna_7b_v1",
        }

        keys_and_defaults = [
            ("context_size", 4096),
            ("prompt_mode", PromptMode.RANK_GPT),
            ("num_few_shot_examples", 0),
            ("device", "cuda"),
            ("num_gpus", 1),
            ("variable_passages", False),
            ("window_size", 20),
            ("system_message", None),
            ("vllm_batched", False),
        ]
        [
            context_size,
            prompt_mode,
            num_few_shot_examples,
            device,
            num_gpus,
            variable_passages,
            window_size,
            system_message,
            vllm_batched,
        ] = extract_kwargs(keys_and_defaults, **kwargs)

        agent = RankListwiseOSLLM(
            model=model_full_paths[model_path]
            if model_path in model_full_paths
            else model_path,
            name=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
            system_message=system_message,
            vllm_batched=vllm_batched,
        )

        print(f"Completed loading {model_path}")
    elif model_path in ["unspecified", "rank_random", "rank_identity"]:
        # NULL reranker
        agent = None
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    return agent


def extract_kwargs(
    keys_and_defaults: List[(str, Any)],
    **kwargs,
):
    extracted_kwargs = [
        kwargs.get(key_and_default[0], key_and_default[-1])
        for key_and_default in keys_and_defaults
    ]

    # Check that type of provided kwarg is compatible with the provided default type
    for i, extracted_kwarg in enumerate(extract_kwargs):
        if type(keys_and_defaults[i[-1]]) != None and (
            type(extracted_kwarg) != type(keys_and_defaults[i[-1]])
        ):
            raise ValueError(
                "Provided kwarg must be compatible with the argument's default type"
            )

    return extracted_kwargs

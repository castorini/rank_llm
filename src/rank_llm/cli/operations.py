from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.rerank import IdentityReranker, Reranker
from rank_llm.retrieve.retrieval_method import RetrievalMethod
from rank_llm.retrieve.retriever import RetrievalMode
from rank_llm.utils import default_device


@dataclass
class ScriptRerankResult:
    args: dict[str, Any]
    results: list[Result] | Any


def _default_retrieve_and_rerank(*args: Any, **kwargs: Any) -> Any:
    from rank_llm.retrieve_and_rerank import retrieve_and_rerank

    return retrieve_and_rerank(*args, **kwargs)


def run_script_rerank(
    args: Any,
    *,
    parser_error: Callable[[str], None],
    runner: Callable[..., Any] = _default_retrieve_and_rerank,
    device_resolver: Callable[[], str] = default_device,
) -> ScriptRerankResult:
    if args.requests_file and args.retrieval_method:
        parser_error("--retrieval_method must not be used with --requests_file")

    if args.dataset and not args.retrieval_method:
        parser_error("--retrieval_method is required when --dataset is provided")

    top_k_rerank = (
        args.top_k_candidates if args.top_k_rerank == -1 else args.top_k_rerank
    )
    retrieval_mode = (
        RetrievalMode.DATASET if args.dataset else RetrievalMode.CACHED_FILE
    )
    options = {
        "model_path": args.model_path,
        "query": "",
        "batch_size": args.batch_size,
        "dataset": args.dataset,
        "retrieval_mode": retrieval_mode,
        "requests_file": args.requests_file,
        "qrels_file": args.qrels_file,
        "output_jsonl_file": args.output_jsonl_file,
        "output_trec_file": args.output_trec_file,
        "invocations_history_file": args.invocations_history_file,
        "retrieval_method": args.retrieval_method,
        "top_k_retrieve": args.top_k_candidates,
        "top_k_rerank": top_k_rerank,
        "max_queries": args.max_queries,
        "context_size": args.context_size,
        "device": device_resolver(),
        "num_gpus": args.num_gpus,
        "prompt_template_path": (
            Path(args.prompt_template_path) if args.prompt_template_path else None
        ),
        "num_few_shot_examples": args.num_few_shot_examples,
        "few_shot_file": args.few_shot_file,
        "shuffle_candidates": args.shuffle_candidates,
        "print_prompts_responses": args.print_prompts_responses,
        "use_azure_openai": args.use_azure_openai,
        "use_openrouter": args.use_openrouter,
        "base_url": args.base_url,
        "variable_passages": args.variable_passages,
        "num_passes": args.num_passes,
        "window_size": args.window_size,
        "stride": args.stride,
        "system_message": args.system_message,
        "populate_invocations_history": args.populate_invocations_history,
        "is_thinking": args.is_thinking,
        "reasoning_token_budget": args.reasoning_token_budget,
        "use_logits": args.use_logits,
        "use_alpha": args.use_alpha,
        "sglang_batched": args.sglang_batched,
        "tensorrt_batched": args.tensorrt_batched,
        "reasoning_effort": args.reasoning_effort,
        "max_passage_words": args.max_passage_words,
    }
    return ScriptRerankResult(args=options, results=runner(**options))


def run_mcp_rerank(
    *,
    model_path: str,
    query_text: str,
    candidates: list[dict[str, Any]],
    query_id: str | int = "",
    batch_size: int = 32,
    top_k_rerank: int = -1,
    context_size: int = 4096,
    num_gpus: int = 1,
    prompt_template_path: str = "",
    num_few_shot_examples: int = 0,
    few_shot_file: str = "",
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    use_azure_openai: bool = False,
    use_openrouter: bool = False,
    base_url: str = "",
    variable_passages: bool = False,
    num_passes: int = 1,
    window_size: int = 20,
    stride: int = 10,
    system_message: str = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
    populate_invocations_history: bool = False,
    is_thinking: bool = False,
    reasoning_token_budget: int = 10000,
    use_logits: bool = False,
    use_alpha: bool = False,
    sglang_batched: bool = False,
    tensorrt_batched: bool = False,
) -> list[Result]:
    kwargs = locals().copy()
    del kwargs["model_path"]
    kwargs["prompt_template_path"] = prompt_template_path or None
    kwargs["few_shot_file"] = few_shot_file or None
    kwargs["base_url"] = base_url or None

    reranker = Reranker(
        Reranker.create_model_coordinator(
            model_path,
            None,
            False,
            **kwargs,
        )
    )

    top_k_retrieve = len(candidates)
    top_k_rerank_effective = top_k_retrieve if top_k_rerank == -1 else top_k_rerank
    del kwargs["top_k_rerank"], kwargs["shuffle_candidates"]
    requests = [
        Request(
            query=Query(text=query_text, qid=query_id),
            candidates=[
                Candidate(
                    docid=c["docid"],
                    score=c["score"],
                    doc={"contents": c["doc"]}
                    if isinstance(c["doc"], str)
                    else c["doc"],
                )
                for c in candidates
            ],
        )
    ]
    if reranker.get_model_coordinator() is None:
        shuffle_candidates = model_path == "rank_random"
        rerank_results = IdentityReranker().rerank_batch(
            requests,
            rank_end=top_k_retrieve,
            shuffle_candidates=shuffle_candidates,
        )
    else:
        for _ in range(num_passes):
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

    for rerank_result in rerank_results:
        rerank_result.candidates = rerank_result.candidates[:top_k_rerank_effective]
    return rerank_results


def run_mcp_retrieve_and_rerank(
    *,
    model_path: str,
    query: str = "",
    batch_size: int = 32,
    dataset: str = "",
    requests_file: str = "",
    qrels_file: str = "",
    output_jsonl_file: str = "",
    output_trec_file: str = "",
    invocations_history_file: str = "",
    retrieval_method: RetrievalMethod = RetrievalMethod.UNSPECIFIED,
    top_k_candidates: int = 100,
    top_k_rerank: int = -1,
    max_queries: int = -1,
    context_size: int = 4096,
    num_gpus: int = 1,
    prompt_template_path: str = "",
    num_few_shot_examples: int = 0,
    few_shot_file: str = "",
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    use_azure_openai: bool = False,
    use_openrouter: bool = False,
    base_url: str = "",
    variable_passages: bool = False,
    num_passes: int = 1,
    window_size: int = 20,
    stride: int = 10,
    system_message: str = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
    populate_invocations_history: bool = False,
    is_thinking: bool = False,
    reasoning_token_budget: int = 10000,
    use_logits: bool = False,
    use_alpha: bool = False,
    sglang_batched: bool = False,
    tensorrt_batched: bool = False,
    runner: Callable[..., Any] = _default_retrieve_and_rerank,
    device_resolver: Callable[[], str] = default_device,
) -> list[Result] | Any:
    top_k_rerank = top_k_candidates if top_k_rerank == -1 else top_k_rerank
    retrieval_mode = RetrievalMode.DATASET if dataset else RetrievalMode.CACHED_FILE
    dataset_or_none = dataset or None
    retrieval_method_or_none = (
        retrieval_method if retrieval_method != RetrievalMethod.UNSPECIFIED else None
    )
    max_queries_or_none = max_queries if max_queries >= 0 else None

    if requests_file and retrieval_method != RetrievalMethod.UNSPECIFIED:
        raise ValueError("retrieval_method must not be used with requests_file")
    if dataset_or_none and not retrieval_method_or_none:
        raise ValueError("retrieval_method is required when dataset is provided")

    return runner(
        model_path=model_path,
        query=query,
        batch_size=batch_size,
        dataset=dataset_or_none,
        retrieval_mode=retrieval_mode,
        requests_file=requests_file,
        qrels_file=qrels_file,
        output_jsonl_file=output_jsonl_file,
        output_trec_file=output_trec_file,
        invocations_history_file=invocations_history_file,
        retrieval_method=retrieval_method_or_none,
        top_k_retrieve=top_k_candidates,
        top_k_rerank=top_k_rerank,
        max_queries=max_queries_or_none,
        context_size=context_size,
        device=device_resolver(),
        num_gpus=num_gpus,
        prompt_template_path=prompt_template_path or None,
        num_few_shot_examples=num_few_shot_examples,
        few_shot_file=few_shot_file or None,
        shuffle_candidates=shuffle_candidates,
        print_prompts_responses=print_prompts_responses,
        use_azure_openai=use_azure_openai,
        use_openrouter=use_openrouter,
        base_url=base_url or None,
        variable_passages=variable_passages,
        num_passes=num_passes,
        window_size=window_size,
        stride=stride,
        system_message=system_message,
        populate_invocations_history=populate_invocations_history,
        is_thinking=is_thinking,
        reasoning_token_budget=reasoning_token_budget,
        use_logits=use_logits,
        use_alpha=use_alpha,
        sglang_batched=sglang_batched,
        tensorrt_batched=tensorrt_batched,
    )

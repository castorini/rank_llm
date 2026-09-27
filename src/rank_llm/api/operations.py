"""Execution helpers used by RankLLM's public interfaces.

Operation-to-interface map:

* ``run_rerank``: CLI ``rerank --input-json/--stdin``, REST
  ``POST /v1/rerank``, and MCP ``rerank``. Reranks supplied candidates.
* ``run_retrieve_and_rerank``: CLI ``rerank --dataset/--requests-file``, REST
  ``POST /v1/retrieve-and-rerank``, and MCP ``retrieve_and_rerank``. Supports
  local datasets, request files, and Pyserini HTTP retrieval via retriever_host.
* ``run_evaluate_aggregate``: CLI ``evaluate`` only.
* ``run_response_analysis_files``: CLI ``analyze`` only.
* ``run_retrieve_cache_generation``: CLI ``retrieve-cache`` only.

The reranking helpers accept RerankOptions and RetrievalOptions objects, built
by each interface. Input checks run before execution. Existing CLI validation
and dry-run commands stop before these helpers; REST and MCP expose execution
only.

These helpers return results or summaries. Interfaces handle their own response
envelopes: CLI JSON and REST use CommandResponse; MCP execution returns results.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from rank_llm.api.options import (
    RerankOptions,
    RetrievalOptions,
    validate_rerank_options,
    validate_retrieval_options,
)
from rank_llm.data import Candidate, Query, Request, Result, normalize_rerank_input
from rank_llm.rerank import Reranker
from rank_llm.retrieve.retrieval_method import RetrievalMethod
from rank_llm.retrieve.retriever import RetrievalMode
from rank_llm.utils import default_device


def run_rerank(
    *,
    options: RerankOptions,
    query_text: str,
    candidates: list[str | dict[str, Any]],
    query_id: str | int = "",
    reranker: Reranker | None = None,
) -> list[Result]:
    """Rerank supplied candidates for CLI, REST, and MCP; return ranked records.

    An injected reranker lets REST reuse its model cache. Otherwise this helper
    initializes the model, runs the requested passes, and truncates the results.
    """
    validate_rerank_options(options)
    kwargs = options.to_kwargs()
    normalized = normalize_rerank_input(
        {"query": {"text": query_text, "qid": query_id}, "candidates": candidates}
    )
    candidates = normalized["candidates"]
    del kwargs["model_path"]

    if reranker is None:
        reranker = Reranker(
            Reranker.create_model_coordinator(
                options.model_path,
                None,
                False,
                device=default_device(),
                **kwargs,
            )
        )

    top_k_retrieve = len(candidates)
    top_k_rerank_effective = (
        top_k_retrieve if options.top_k_rerank == -1 else options.top_k_rerank
    )
    kwargs["top_k_rerank"] = top_k_rerank_effective
    requests = [
        Request(
            query=Query(text=query_text, qid=query_id),
            candidates=[Candidate(**c) for c in candidates],
        )
    ]
    return reranker.rerank_passes(
        requests, model_path=options.model_path, top_k_retrieve=top_k_retrieve, **kwargs
    )


def run_retrieve_and_rerank(
    *,
    options: RerankOptions,
    retrieval: RetrievalOptions,
    reranker: Reranker | None = None,
    runner: Callable[..., Any] | None = None,
    device_resolver: Callable[[], str] = default_device,
) -> list[Result]:
    """Run dataset, file, or HTTP-service retrieval and reranking for all interfaces.

    Translate the option objects to pipeline arguments. The existing retrieval
    pipeline owns output files and history writes. An injected reranker supports
    REST's model cache.
    """
    validate_rerank_options(options)
    validate_retrieval_options(retrieval, rerank_options=options)
    method = RetrievalMethod(retrieval.retrieval_method or "unspecified")
    kwargs = {**options.to_kwargs(), **asdict(retrieval)}
    kwargs["qid"] = kwargs.pop("query_id")
    kwargs["top_k_retrieve"] = kwargs.pop("top_k_candidates")
    kwargs["top_k_rerank"] = (
        retrieval.top_k_candidates
        if options.top_k_rerank == -1
        else options.top_k_rerank
    )
    kwargs["retrieval_mode"] = (
        RetrievalMode.DATASET if retrieval.dataset else RetrievalMode.CACHED_FILE
    )
    kwargs["dataset"] = retrieval.dataset or None
    kwargs["retrieval_method"] = (
        method if method != RetrievalMethod.UNSPECIFIED else None
    )
    kwargs["max_queries"] = (
        retrieval.max_queries if retrieval.max_queries >= 0 else None
    )
    if runner is None:
        from rank_llm.retrieve_and_rerank import retrieve_and_rerank

        runner = retrieve_and_rerank
    return runner(**kwargs, reranker=reranker, device=device_resolver())


def run_evaluate_aggregate(
    *,
    model_name: str,
    context_size: int = 4096,
    rerank_results_dirname: str = "rerank_results",
    runner: Callable[[str, int, str], Any] | None = None,
) -> dict[str, Any]:
    """Support CLI ``evaluate``; no REST route or RankLLM MCP tool exposes this."""
    if runner is None:
        from argparse import Namespace

        from rank_llm.scripts.run_trec_eval import evaluate_aggregate

        args = Namespace(
            model_name=model_name,
            context_size=context_size,
            rerank_results_dirname=rerank_results_dirname,
        )
        evaluate_aggregate(args)
    else:
        runner(model_name, context_size, rerank_results_dirname)
    return {
        "model_name": model_name,
        "context_size": context_size,
        "rerank_results_dirname": rerank_results_dirname,
        "output_file": f"trec_eval_aggregated_results_{model_name}.jsonl",
    }


def run_response_analysis_files(
    *,
    files: list[str],
    verbose: bool = False,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Support CLI ``analyze``; no REST route or RankLLM MCP tool exposes this."""
    if runner is None:
        from rank_llm.analysis.response_analysis import ResponseAnalyzer

        runner = ResponseAnalyzer.from_stored_files
        analyzer = runner(files)
        return {
            "files": files,
            "verbose": verbose,
            "metrics": analyzer.count_errors(verbose),
        }
    return runner(files, verbose)


def run_retrieve_cache_generation(
    *,
    trec_file: str,
    collection_file: str,
    query_file: str,
    output_file: str,
    output_trec_file: str | None = None,
    topk: int = 20,
    generator: Callable[..., Any] | None = None,
    writer: Callable[[str, Any], None] | None = None,
) -> dict[str, Any]:
    """Support CLI ``retrieve-cache``; no REST route or RankLLM MCP tool exposes this."""
    if generator is None or writer is None:
        from rank_llm.scripts.generate_retrieve_results_json_cache import (
            generate_retrieve_results,
            write_output_file,
        )

        generator = generator or generate_retrieve_results
        writer = writer or write_output_file
    results = generator(
        trec_file,
        collection_file,
        query_file,
        topk,
        output_trec_file,
    )
    writer(output_file, results)
    return {
        "trec_file": trec_file,
        "collection_file": collection_file,
        "query_file": query_file,
        "output_file": output_file,
        "output_trec_file": output_trec_file,
        "topk": topk,
        "record_count": len(results),
    }

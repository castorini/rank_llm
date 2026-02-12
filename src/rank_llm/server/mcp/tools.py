"""
Register tools for the MCP server.

All parameters use explicit types and sentinel defaults (no Optional[X]) so that
the generated JSON Schema has a single "type" per property. This avoids vLLM's
trim_schema failing on anyOf entries that lack a "type" key.
"""

import copy
from typing import Any

import torch
from fastmcp import FastMCP

from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.rerank import IdentityReranker, Reranker
from rank_llm.retrieve import TOPICS, RetrievalMethod, RetrievalMode
from rank_llm.retrieve_and_rerank import (
    retrieve_and_rerank as retrieve_and_rerank_function,
)


def register_rankllm_tools(mcp: FastMCP):
    """Register RankLLM tools with the MCP server."""

    @mcp.tool(
        description=f"""
        Rerank retrieval results using the specified model and parameters.
        Use this only when you need to rerank a small number of given candidates.

        Args:
            model_path: Path to the model. If `use_azure_ai`, pass your deployment name.
            batch_size: Size of each batch for batched inference.
            query_text: Query text to get search results for.
            candidates: List of candidates to rerank.
            query_id: Query ID.
            top_k_rerank: the number of top candidates to return from reranking (-1 means same as top_k_candidates)
            context_size: context size used for model
            num_gpus: the number of GPUs to use
            prompt_template_path: yaml file path for the prompt template
            num_few_shot_examples: number of in context examples to provide
            few_shot_file: path to JSONL file containing few-shot examples.
            shuffle_candidates: whether to shuffle the candidates before reranking.
            print_prompts_responses: whether to print prompts and responses.
            use_azure_openai: If True, use Azure OpenAI. Requires env var to be set: `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`.
            use_openrouter: If True, use OpenRouter. Requires env var to be set: `OPENROUTER_API_KEY`
            base_url: If not using OpenAI's endpoint, pass your base URL and provide API key. Requires env var to be set: `OPENAI_API_KEY`
            variable_passages: whether the model can account for variable number of passages in input.
            num_passes: number of passes to run the model
            window_size: window size for the sliding window approach.
            stride: stride for the sliding window approach.
            system_message: the system message used in prompts.
            populate_invocations_history: write a file with the prompts and raw responses from LLM.
            is_thinking: enables thinking mode which increases output token budget to account for the full thinking trace + response.
            reasoning_token_budget: number of output token budget for thinking traces on reasoning models.
            use_logits: whether to rerank using the logits of the first identifier only.
            use_alpha: whether to use alphabetical identifers instead of numerical. Recommended when use_logits is True.
            sglang_batched: whether to run the model in batches using sglang backend.
            tensorrt_batched: whether to run the model in batches using tensorrtllm backend.
        """
    )
    def rerank(
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

        # Convert sentinel defaults to None for Reranker.create_model_coordinator (reranker.py: extract_kwargs with None defaults)
        prompt_template_path_or_none = (
            prompt_template_path if prompt_template_path else None
        )
        few_shot_file_or_none = few_shot_file if few_shot_file else None
        base_url_or_none = base_url if base_url else None
        kwargs["prompt_template_path"] = prompt_template_path_or_none
        kwargs["few_shot_file"] = few_shot_file_or_none
        kwargs["base_url"] = base_url_or_none

        reranker = Reranker(
            Reranker.create_model_coordinator(
                model_path,
                None,
                False,
                **kwargs,
            )
        )

        top_k_retrieve = len(candidates)
        # -1 means same as input size (reranker.rerank_batch / final slice)
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
                        if type(c["doc"]) is str
                        else c["doc"],
                    )
                    for c in candidates
                ],
            )
        ]
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
            rr.candidates = rr.candidates[:top_k_rerank_effective]

        return rerank_results

    @mcp.tool(
        description=f"""
        Rerank retrieval results using the specified model and parameters.
        Use this most of the time to conserve context window.

        Args:
            model_path: Path to the model. If `use_azure_ai`, pass your deployment name.
            query: Query text to get search results for.
            batch_size: Size of each batch for batched inference.
            dataset: Should be one of 1- dataset name, must be in {TOPICS.keys()},  2- a list of inline documents  3- a list of inline hits; must be used when --requests_file is not specified
            retrieval_mode: Mode of retrieval, either {RetrievalMode.DATASET} or {RetrievalMode.CACHED_FILE}.
            requests_file: Path to a JSONL file containing requests; must be used when --dataset is not specified.
            qrels_file: Optional. With --dataset: override default qrels. With --requests_file: qrels file for Trec eval
            output_jsonl_file: Optional. With --dataset: override computed JSONL output path. With --requests_file: required path where ranked results are saved
            output_trec_file:Optional. With --dataset: override computed TREC output path. With --requests_file: required path where ranked results are saved (trec format)
            invocations_history_file: Optional. With --dataset: override computed invocations history path. With --requests_file and --populate_invocations_history: required path for LLM invocations history (prompts, completions, and input/output token counts)
            retrieval_method: Required when dataset is provided; use "unspecified" when using requests_file. One of: {[m.value for m in RetrievalMethod]}.
            top_k_candidates: the number of top candidates to rerank
            top_k_rerank: the number of top candidates to return from reranking (-1 means same as top_k_candidates)
            max_queries: max number of queries to process (-1 means no limit)
            context_size: context size used for model
            num_gpus: the number of GPUs to use
            prompt_template_path: yaml file path for the prompt template
            num_few_shot_examples: number of in context examples to provide
            few_shot_file: path to JSONL file containing few-shot examples.
            shuffle_candidates: whether to shuffle the candidates before reranking.
            print_prompts_responses: whether to print prompts and responses.
            use_azure_openai: If True, use Azure OpenAI. Requires env var to be set: `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`.
            use_openrouter: If True, use OpenRouter. Requires env var to be set: `OPENROUTER_API_KEY`
            base_url: If not using OpenAI's endpoint, pass your base URL and provide API key. Requires env var to be set: `OPENAI_API_KEY`
            variable_passages: whether the model can account for variable number of passages in input.
            num_passes: number of passes to run the model
            window_size: window size for the sliding window approach.
            stride: stride for the sliding window approach.
            system_message: the system message used in prompts.
            populate_invocations_history: write a file with the prompts and raw responses from LLM.
            is_thinking: enables thinking mode which increases output token budget to account for the full thinking trace + response.
            reasoning_token_budget: number of output token budget for thinking traces on reasoning models.
            use_logits: whether to rerank using the logits of the first identifier only.
            use_alpha: whether to use alphabetical identifers instead of numerical. Recommended when use_logits is True.
            sglang_batched: whether to run the model in batches using sglang backend.
            tensorrt_batched: whether to run the model in batches using tensorrtllm backend.
        """
    )
    def retrieve_and_rerank(
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
    ) -> list[Result]:
        top_k_rerank = top_k_candidates if top_k_rerank == -1 else top_k_rerank
        device = "cuda" if torch.cuda.is_available() else "cpu"
        retrieval_mode = RetrievalMode.DATASET if dataset else RetrievalMode.CACHED_FILE

        # Convert sentinel defaults to None for the underlying API
        dataset_or_none = dataset if dataset else None
        retrieval_method_or_none = (
            retrieval_method
            if retrieval_method != RetrievalMethod.UNSPECIFIED
            else None
        )
        max_queries_or_none = max_queries if max_queries >= 0 else None
        prompt_template_path_or_none = (
            prompt_template_path if prompt_template_path else None
        )
        few_shot_file_or_none = few_shot_file if few_shot_file else None
        base_url_or_none = base_url if base_url else None

        if requests_file:
            if retrieval_method != RetrievalMethod.UNSPECIFIED:
                raise ValueError("retrieval_method must not be used with requests_file")
        if dataset_or_none and not retrieval_method_or_none:
            raise ValueError("retrieval_method is required when dataset is provided")

        return retrieve_and_rerank_function(
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
            device=device,
            num_gpus=num_gpus,
            prompt_template_path=prompt_template_path_or_none,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file_or_none,
            shuffle_candidates=shuffle_candidates,
            print_prompts_responses=print_prompts_responses,
            use_azure_openai=use_azure_openai,
            use_openrouter=use_openrouter,
            base_url=base_url_or_none,
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

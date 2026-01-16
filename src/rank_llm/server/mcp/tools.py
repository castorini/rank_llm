"""
Register tools for the MCP server.
"""

import torch
from fastmcp import FastMCP
from pyserini.server.mcp.tools import register_tools
from pyserini.server.search_controller import get_controller

from rank_llm.data import Result
from rank_llm.retrieve import TOPICS, RetrievalMethod, RetrievalMode
from rank_llm.retrieve_and_rerank import (
    retrieve_and_rerank as retrieve_and_rerank_function,
)


def register_rankllm_tools(mcp: FastMCP):
    """Register all tools with the MCP server."""

    register_tools(mcp, get_controller())

    @mcp.tool(
        name="retrieve_and_rerank",
        description=f"""
        Rerank retrieval results using the specified model and parameters.

        Args:
            model_path: Path to the model. If `use_azure_ai`, pass your deployment name.
            query: Query text to get search results for.
            batch_size: Size of each batch for batched inference.
            dataset: Should be one of 1- dataset name, must be in {TOPICS.keys()},  2- a list of inline documents  3- a list of inline hits; must be used when --requests_file is not specified
            retrieval_mode: Mode of retrieval, either {RetrievalMode.DATASET} or {RetrievalMode.CACHED_FILE}.
            requests_file: Path to a JSONL file containing requests; must be used when --dataset is not specified.
            qrels_file: Only used with --requests_file; when present the Trec eval will be executed using this qrels file
            output_jsonl_file: Only used with --requests_file; when present, the ranked results will be saved in this JSONL file.
            output_trec_file: Only used with --requests_file; when present, the ranked results will be saved in this txt file in trec format.
            invocations_history_file: Only used with --requests_file and --populate_invocations_history; when present, the LLM invocations history (prompts, completions, and input/output token counts) will be stored in this file.
            retrieval_method: Required if --dataset is used; must be omitted with --requests_file
            top_k_candidates: the number of top candidates to rerank
            top_k_rerank: the number of top candidates to return from reranking
            max_queries: the max number of queries to process from the dataset
            context_size: context size used for model
            num_gpus: the number of GPUs to use
            prompt_template_path: yaml file path for the prompt template
            num_few_shot_examples: number of in context examples to provide
            few_shot_file: path to JSONL file containing few-shot examples.
            shuffle_candidates: whether to shuffle the candidates before reranking.
            print_prompts_responses: whether to print prompts and responses.
            use_azure_openai: If True, use Azure OpenAI. Requires env var to be set: `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`.
            use_openrouter: If True, use OpenRouter. Requires env var to be set: `OPENROUTER_API_KEY`
            base_url: If using a non-OpenAI model, pass your base URL and provide API key. Requires env var to be set: `OPENAI_API_KEY`
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
        """,
    )
    def retrieve_and_rerank(
        model_path: str,
        query: str = "",
        batch_size: int = 32,
        dataset: str = None,
        requests_file: str = "",
        qrels_file: str = "",
        output_jsonl_file: str = "",
        output_trec_file: str = "",
        invocations_history_file: str = "",
        retrieval_method: RetrievalMethod = None,
        top_k_candidates: int = 100,
        top_k_rerank: int = -1,
        max_queries: int = None,
        context_size: int = 4096,
        num_gpus: int = 1,
        prompt_template_path: str = None,
        num_few_shot_examples: int = 0,
        few_shot_file: str = None,
        shuffle_candidates: bool = False,
        print_prompts_responses: bool = False,
        use_azure_openai: bool = False,
        use_openrouter: bool = False,
        base_url: str = None,
        variable_passages: bool = False,
        num_passes: int = 1,
        window_size: int = 20,
        stride: str = 10,
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

        if requests_file:
            if retrieval_method:
                raise ValueError("retrieval_method must not be used with requests_file")
        if dataset and not retrieval_method:
            raise ValueError("retrieval_method is required when dataset is provided")

        return retrieve_and_rerank_function(
            model_path=model_path,
            query=query,
            batch_size=batch_size,
            dataset=dataset,
            retrieval_mode=retrieval_mode,
            requests_file=requests_file,
            qrels_file=qrels_file,
            output_jsonl_file=output_jsonl_file,
            output_trec_file=output_trec_file,
            invocations_history_file=invocations_history_file,
            retrieval_method=retrieval_method,
            top_k_retrieve=top_k_candidates,
            top_k_rerank=top_k_rerank,
            max_queries=max_queries,
            context_size=context_size,
            device=device,
            num_gpus=num_gpus,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            shuffle_candidates=shuffle_candidates,
            print_prompts_responses=print_prompts_responses,
            use_azure_openai=use_azure_openai,
            use_openrouter=use_openrouter,
            base_url=base_url,
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

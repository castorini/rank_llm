"""Register tools for the MCP server."""

from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field, StrictBool, StrictInt, StrictStr

from rank_llm.api.operations import run_rerank, run_retrieve_and_rerank
from rank_llm.api.options import (
    RerankOptions,
    RetrievalOptions,
    option_schema,
    option_values,
)
from rank_llm.data import Result
from rank_llm.retrieve import RetrievalMethod


def _with_option_schema(fn):
    """Keep explicit tool signatures while publishing the shared constraints."""
    properties = {
        **option_schema()["properties"],
        **option_schema(RetrievalOptions)["properties"],
    }
    for name, annotation in fn.__annotations__.items():
        if name in properties:
            schema = {
                key: value
                for key, value in properties[name].items()
                if key != "default"
            }
            fn.__annotations__[name] = Annotated[
                annotation, Field(json_schema_extra=schema)
            ]
    return fn


def register_rankllm_tools(mcp: FastMCP):
    """Register RankLLM tools with the MCP server."""

    @mcp.tool(
        description="""
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
            use_alpha: whether to use alphabetical identifiers instead of numerical. Recommended when use_logits is True.
            use_litellm: Route model calls through LiteLLM.
            pointwise_vllm: Use the pointwise vLLM backend.
            listwise_vllm_with_openai_sdk: Use remote listwise vLLM via base_url.
            reasoning_effort: One of none, minimal, low, medium, high, xhigh; null uses the backend default.
            max_passage_words: Maximum words per passage.
        """
    )
    @_with_option_schema
    def rerank(
        model_path: StrictStr,
        query_text: StrictStr,
        candidates: list[str | dict[str, Any]],
        query_id: StrictStr | StrictInt = "",
        batch_size: StrictInt = 32,
        top_k_rerank: StrictInt = -1,
        context_size: StrictInt = 4096,
        num_gpus: StrictInt = 1,
        prompt_template_path: StrictStr = "",
        num_few_shot_examples: StrictInt = 0,
        few_shot_file: StrictStr = "",
        shuffle_candidates: StrictBool = False,
        print_prompts_responses: StrictBool = False,
        use_azure_openai: StrictBool = False,
        use_openrouter: StrictBool = False,
        use_litellm: StrictBool = False,
        base_url: StrictStr = "",
        variable_passages: StrictBool = False,
        num_passes: StrictInt = 1,
        window_size: StrictInt = 20,
        stride: StrictInt = 10,
        system_message: StrictStr = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
        populate_invocations_history: StrictBool = False,
        is_thinking: StrictBool = False,
        reasoning_token_budget: StrictInt = 10000,
        use_logits: StrictBool = False,
        use_alpha: StrictBool = False,
        pointwise_vllm: StrictBool = False,
        listwise_vllm_with_openai_sdk: StrictBool = False,
        reasoning_effort: StrictStr | None = None,
        max_passage_words: StrictInt = 300,
    ) -> list[Result]:
        return run_rerank(
            options=RerankOptions(**option_values(locals())),
            query_text=query_text,
            query_id=query_id,
            candidates=candidates,
        )

    @mcp.tool(
        description="""
        Rerank retrieval results using the specified model and parameters.
        Use this most of the time to conserve context window.

        Args:
            model_path: Path to the model. If `use_azure_ai`, pass your deployment name.
            query: Query text to get search results for.
            query_id: Query ID for a supplied query (default 1).
            retriever_host: Optional Pyserini HTTP base URL; requires a dataset, nonempty query, and BM25.
            batch_size: Size of each batch for batched inference.
            dataset: Dataset or index name. Mutually exclusive with requests_file.
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
            use_alpha: whether to use alphabetical identifiers instead of numerical. Recommended when use_logits is True.
            use_litellm: Route model calls through LiteLLM.
            pointwise_vllm: Use the pointwise vLLM backend.
            listwise_vllm_with_openai_sdk: Use remote listwise vLLM via base_url.
            reasoning_effort: One of none, minimal, low, medium, high, xhigh; null uses the backend default.
            max_passage_words: Maximum words per passage.
        """
    )
    @_with_option_schema
    def retrieve_and_rerank(
        model_path: StrictStr,
        query: StrictStr = "",
        query_id: StrictStr | StrictInt = 1,
        retriever_host: StrictStr = "",
        batch_size: StrictInt = 32,
        dataset: StrictStr = "",
        requests_file: StrictStr = "",
        qrels_file: StrictStr = "",
        output_jsonl_file: StrictStr = "",
        output_trec_file: StrictStr = "",
        invocations_history_file: StrictStr = "",
        retrieval_method: RetrievalMethod = RetrievalMethod.UNSPECIFIED,
        top_k_candidates: StrictInt = 100,
        top_k_rerank: StrictInt = -1,
        max_queries: StrictInt = -1,
        context_size: StrictInt = 4096,
        num_gpus: StrictInt = 1,
        prompt_template_path: StrictStr = "",
        num_few_shot_examples: StrictInt = 0,
        few_shot_file: StrictStr = "",
        shuffle_candidates: StrictBool = False,
        print_prompts_responses: StrictBool = False,
        use_azure_openai: StrictBool = False,
        use_openrouter: StrictBool = False,
        use_litellm: StrictBool = False,
        base_url: StrictStr = "",
        variable_passages: StrictBool = False,
        num_passes: StrictInt = 1,
        window_size: StrictInt = 20,
        stride: StrictInt = 10,
        system_message: StrictStr = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
        populate_invocations_history: StrictBool = False,
        is_thinking: StrictBool = False,
        reasoning_token_budget: StrictInt = 10000,
        use_logits: StrictBool = False,
        use_alpha: StrictBool = False,
        pointwise_vllm: StrictBool = False,
        listwise_vllm_with_openai_sdk: StrictBool = False,
        reasoning_effort: StrictStr | None = None,
        max_passage_words: StrictInt = 300,
    ) -> list[Result]:
        return run_retrieve_and_rerank(
            options=RerankOptions(**option_values(locals())),
            retrieval=RetrievalOptions(**option_values(locals(), RetrievalOptions)),
        )

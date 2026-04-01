"""Register tools for the MCP server."""

from typing import Any

from fastmcp import FastMCP

from rank_llm.cli.operations import run_mcp_rerank, run_mcp_retrieve_and_rerank
from rank_llm.data import Result
from rank_llm.retrieve import RetrievalMethod


def register_rankllm_tools(mcp: FastMCP[Any]) -> None:
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
        return run_mcp_rerank(**locals())

    @mcp.tool(
        description="""
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
        return run_mcp_retrieve_and_rerank(**locals())

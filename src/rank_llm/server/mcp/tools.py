"""
Register tools for the MCP server.
"""

from typing import Any

from fastmcp import FastMCP
from pyserini.server.mcp.tools import register_tools
from pyserini.server.search_controller import get_controller

from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.rerank import IdentityReranker, Reranker
from rank_llm.rerank.rankllm import PromptMode


def register_rankllm_tools(mcp: FastMCP):
    """Register all tools with the MCP server."""

    register_tools(mcp, get_controller())

    @mcp.tool(
        name="rerank",
        description="Reranks retrieval results with given model and arguments.",
    )
    def rerank(
        model_path: str,
        query_text: str,
        candidates: list[dict[str, Any]],
        query_id: str | int = "",
        top_k_rerank: int = 10,
        shuffle_candidates: bool = False,
        print_prompts_responses: bool = False,
        num_passes: int = 1,
        batch_size: int = 32,
        use_azure_openai: bool = False,
        context_size: int = 4096,
        num_gpus: int = 1,
        prompt_mode: PromptMode = None,
        prompt_template_path: str = None,
        num_few_shot_examples: int = 0,
        few_shot_file: str = None,
        variable_passages: bool = False,
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
        f"""
        Rerank retrieval results using the specified model and parameters.

        Args:
            model_path: Path to the model. If `use_azure_ai`, pass your deployment name.
            query_text: The query text to rerank candidates for.
            candidates: List of candidate documents to rerank, in the format of {{docid: 'doc_id', 'score': 0.0, doc: {{contents: 'document contents'}}}}.
            query_id: Optional query identifier. 
            top_k_rerank: the number of top candidates to return from reranking.
            shuffle_candidates: whether to shuffle the candidates before reranking.
            print_prompts_responses: whether to print promps and responses.
            num_passes: number of passes to run the model
            batch_size: Size of each batch for batched inference.
            use_azure_openai: If True, use Azure OpenAI. Requires env var to be set: `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`.
            context_size: context size used for model.
            num_gpus: the number of GPUs to use.
            prompt_mode: the prompt mode to use, options are {list(PromptMode)}.
            prompt_template_path: yaml file path for the prompt template.
            num_few_shot_examples: number of in context examples to provide.
            few_shot_file: path to JSONL file containing few-shot examples.
            variable_passages: whether the model can account for variable number of passages in input.
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
        kwargs = locals().copy()
        del kwargs["model_path"]
        reranker = Reranker(
            Reranker.create_model_coordinator(
                model_path,
                None,
                False,
                **kwargs,
            )
        )

        top_k_retrieve = len(candidates)
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
            rr.candidates = rr.candidates[:top_k_rerank]

        return rerank_results

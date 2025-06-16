from typing import Any, List, Optional

from rank_llm.data import Request, Result
from rank_llm.rerank import PromptMode
from rank_llm.rerank.listwise import RankListwiseOSLLM


class QwenReranker:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        prompt_template_path: Optional[str] = None,
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = True,
        window_size: int = 20,
        #is_thinking: bool = True,
        #reasoning_token_budget: int = 10000000000000,
        system_message: str = """You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.
Given a query and a list of passages, your task is to re-rank these passages based on their relevance to the query.

Please perform the following steps:
1. **Understand the Query**: First, carefully read and understand the user's query to identify the core information need.
2. **Analyze Each Passage**: For each passage, critically evaluate its content and determine how well it addresses the query. Consider factors like:
   - Directness of the answer
   - Completeness of the information
   - Presence of supporting evidence or details
   - Absence of irrelevant or distracting information
3. **Compare and Contrast**: Compare the passages against each other. Identify which passages are more relevant and why. Note any subtle differences in relevance.
4. **Reasoning for Ranking**: Explicitly state your reasoning for the rank you assign to each passage. Explain why a passage is ranked higher or lower than others. This step-by-step thought process is crucial.
5. **Assign Ranks**: Based on your analysis and reasoning, assign a unique rank to each passage, starting from 1 for the most relevant.

**Output Format:**
Your final output must be a list of ranks, corresponding to the original order of the passages. For example, if there are 3 passages, and you decide the second passage is most relevant, the first is second most relevant, and the third is least relevant, your output should be:
[2] > [1] > [3]

No other text or explanation should be present in the final output, only the list of ranks.""",
    ) -> None:
        self._reranker = RankListwiseOSLLM(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
            system_message=system_message,
        )

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        """
        Reranks a list of requests using the Qwen model.

        Args:
            requests (List[Request]): The list of requests. Each request has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                window_size (int): The size of the sliding window for listwise reranking, defualts to 20.
                stride (int): The size of the stride of the sliding window for listwise rernaking, defaults to 10.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window size.
        Returns:
            List[Result]: A list containing the reranked results.

        Note:
            check 'reranker.rerank_batch' for implementation details of reranking process.
        """
        return self._reranker.rerank_batch(
            requests=requests,
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            **kwargs,
        )

    def rerank(
        self,
        request: Request,
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> Result:
        """
        Reranks a request using the Qwen model.

        Args:
            request (Request): The reranking request which has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                window_size (int): The size of the sliding window for listwise reranking, defualts to 20.
                stride (int): The size of the stride of the sliding window for listwise rernaking, defaults to 10.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window size.
        Returns:
            Result: the rerank result which contains the reranked candidates.

        Note:
            check 'reranker.rerank' for implementation details of reranking process.
        """
        return self._reranker.rerank_batch(
            requests=[request],
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            **kwargs,
        ) 
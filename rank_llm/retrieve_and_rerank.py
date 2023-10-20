from enum import Enum
from typing import List, Union, Dict, Any

from tqdm import tqdm

from rank_llm.pyserini_retriever import PyseriniRetriever
from rank_llm.rankllm import RankLLM


class RetrievalMode(Enum):
    DATASET = "dataset"
    QUERY_AND_DOCUMENTS = "query_and_documents"
    QUERY_AND_HITS = "query_and_hits"

    def __str__(self):
        return self.value
    

class Retriever:
    def __init__(
        self, retrieval_mode: RetrievalMode
    ) -> None:
        self._retrieval_mode = retrieval_mode

    def retrieve(self, **kwargs) -> Union[None, List[Dict[str, Any]]]:
        '''
        Retriever supports three modes:

        - DATASET: args = (dataset, retrieval_method)
        - QUERY_AND_DOCUMENTS: args = (query, documents)
        - QUERY_AND_HITS: args = (query, hits)
        '''
        if self._retrieval_mode == RetrievalMode.DATASET:
            dataset = kwargs["dataset"]
            retrieval_method = kwargs["retrieval_method"]
            print(f"Retrieving with dataset {dataset}:")
            retriever = PyseriniRetriever(dataset, retrieval_method)
            # Always retrieve top 100 so that results are reusable for all top_k_candidates values.
            retriever.retrieve_and_store(k=100)
            return None
        elif self._retrieval_mode == RetrievalMode.QUERY_AND_DOCUMENTS:
            query = kwargs["dataset"]
            documents = kwargs["documents"]
            document_hits = []
            for passage in documents:
                document_hits.append({
                    "content": passage
                })
            retrieved_result = [{
                "query": query,
                "hits": document_hits,
            }]
            return retrieved_result
        elif self._retrieval_mode == RetrievalMode.QUERY_AND_HITS:
            query = kwargs["dataset"]
            hits = kwargs["hits"]
            retrieved_result = [{
                "query": query,
                "hits": hits,
            }]
            return retrieved_result
        else:
            raise ValueError(f"Invalid retrieval mode: {self._retrieval_mode}")


class Reranker:
    def __init__(
        self, agent: RankLLM
    ) -> None:
        self._agent = agent

    def rerank(self, retrieved_results: List[Dict[str, Any]], **kwargs):
        rerank_results = []
        input_token_counts = []
        output_token_counts = []
        aggregated_prompts = []
        aggregated_responses = []

        for result in tqdm(retrieved_results):
            (
                rerank_result,
                in_token_count,
                out_token_count,
                prompts,
                responses,
            ) = self._agent.sliding_windows(
                result,
                rank_start=0,
                rank_end=kwargs["rank_end"],
                window_size=kwargs["window_size"],
                step=10,
                shuffle_candidates=kwargs["shuffle_candidates"],
                logging=kwargs["logging"],
            )
            rerank_results.append(rerank_result)
            input_token_counts.append(in_token_count)
            output_token_counts.append(out_token_count)
            aggregated_prompts.extend(prompts)
            aggregated_responses.extend(responses)
        
        print(f"rerank_results={rerank_results}")
        print(f"input_tokens_counts={input_token_counts}")
        print(f"total input token count={sum(input_token_counts)}")
        print(f"output_token_counts={output_token_counts}")
        print(f"total output token count={sum(output_token_counts)}")

        return rerank_results, input_token_counts, output_token_counts, aggregated_prompts, aggregated_responses
    
from enum import Enum
from typing import List, Union, Dict, Any
from datetime import datetime
import json
from pathlib import Path

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
            query = kwargs["query"]
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
            query = kwargs["query"]
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
    
    def write_rerank_results(
        self,
        retrieval_method_name: str,
        rerank_results: List[Dict[str, Any]],
        input_token_counts: List[int],
        output_token_counts: List[int],
        # List[str] for Vicuna, List[List[Dict[str, str]]] for gpt models.
        prompts: Union[List[str], List[List[Dict[str, str]]]],
        responses: List[str],
        shuffle_candidates: bool = False,
    ) -> str:
        # write rerank results
        Path(f"../rerank_results/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        _modelname = self._agent._model.split("/")[-1]
        name = f"{_modelname}_{self._agent._context_size}_{self._agent._prompt_mode}"
        name = (
            f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
            if shuffle_candidates
            else f"{name}_{datetime.isoformat(datetime.now())}"
        )
        result_file_name = f"../rerank_results/{retrieval_method_name}/{name}.txt"
        with open(result_file_name, "w") as f:
            for i in range(len(rerank_results)):
                rank = 1
                hits = rerank_results[i]["hits"]
                for hit in hits:
                    f.write(
                        f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n"
                    )
                    rank += 1
        # Write token counts
        Path(f"../token_counts/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        count_file_name = f"../token_counts/{retrieval_method_name}/{name}.txt"
        counts = {}
        for i, (in_count, out_count) in enumerate(
            zip(input_token_counts, output_token_counts)
        ):
            counts[rerank_results[i]["query"]] = (in_count, out_count)
        with open(count_file_name, "w") as f:
            json.dump(counts, f, indent=4)
        # Write prompts and responses
        Path(f"../prompts_and_responses/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            f"../prompts_and_responses/{retrieval_method_name}/{name}.json",
            "w",
        ) as f:
            for p, r in zip(prompts, responses):
                json.dump({"prompt": p, "response": r}, f)
                f.write("\n")
        return result_file_name
    
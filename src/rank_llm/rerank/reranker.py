import json
from datetime import datetime
from pathlib import Path
from typing import List, Union, Dict, Any

from tqdm import tqdm

from rank_llm.rerank.rankllm import RankLLM


class Reranker:
    def __init__(
        self,
        agent: RankLLM,
        top_k_candidates: int,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
    ) -> None:
        self._agent = agent
        self._top_k_candidates = top_k_candidates
        self._dataset = dataset

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
                step=kwargs["step"],
                shuffle_candidates=kwargs["shuffle_candidates"],
                logging=kwargs["logging"],
            )
            rerank_results.append(rerank_result)
            input_token_counts.append(in_token_count)
            output_token_counts.append(out_token_count)
            aggregated_prompts.extend(prompts)
            aggregated_responses.extend(responses)

        # print(f"rerank_results={rerank_results}")
        print(f"input_tokens_counts={input_token_counts}")
        print(f"total input token count={sum(input_token_counts)}")
        print(f"output_token_counts={output_token_counts}")
        print(f"total output token count={sum(output_token_counts)}")

        return (
            rerank_results,
            input_token_counts,
            output_token_counts,
            aggregated_prompts,
            aggregated_responses,
        )

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
        pass_ct: int = None,
        window_size: int = None,
    ) -> str:
        # write rerank results
        Path(f"rerank_results/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        _modelname = self._agent._model.split("/")[-1]
        if _modelname.startswith("checkpoint"):
            _modelname = self._agent._model.split("/")[-2] + "_" + _modelname
        name = f"{_modelname}_{self._agent._context_size}_{self._top_k_candidates}_{self._agent._prompt_mode}_{self._dataset}"
        if self._agent._num_few_shot_examples > 0:
            name += f"_{self._agent._num_few_shot_examples}_shot"
        name = (
            f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
            if shuffle_candidates
            else f"{name}_{datetime.isoformat(datetime.now())}"
        )
        if window_size is not None:
            name += f"_window_{window_size}"
        if pass_ct is not None:
            name += f"_pass_{pass_ct}"
        result_file_name = f"rerank_results/{retrieval_method_name}/{name}.txt"
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
        Path(f"token_counts/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        count_file_name = f"token_counts/{retrieval_method_name}/{name}.txt"
        counts = {}
        for i, (in_count, out_count) in enumerate(
            zip(input_token_counts, output_token_counts)
        ):
            counts[rerank_results[i]["query"]] = (in_count, out_count)
        with open(count_file_name, "w") as f:
            json.dump(counts, f, indent=4)
        # Write prompts and responses
        Path(f"prompts_and_responses/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            f"prompts_and_responses/{retrieval_method_name}/{name}.json",
            "w",
        ) as f:
            for p, r in zip(prompts, responses):
                json.dump({"prompt": p, "response": r}, f)
                f.write("\n")
        return result_file_name

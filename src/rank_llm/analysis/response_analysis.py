import argparse
import json
import os
from typing import List, Dict
import re

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.rerank.rankllm import PromptMode


class ResponseAnalyzer:
    def __init__(
        self,
        model_name: str,
        context_size: int,
        top_candidates: int,
        prompt_mode: PromptMode,
    ) -> None:
        self._model_name = model_name
        self._context_size = context_size
        self._top_candidates = top_candidates
        self._prompt_mode = prompt_mode

    def _get_num_passages(self, prompt) -> int:
        search_text = ""
        if type(prompt) == str:
            search_text = prompt
        # For GPT runs, the prompt is an array of json objects with "role" and "content" as keys.
        elif type(prompt) == list:
            for message in prompt:
                search_text += message["content"]
        else:
            raise ValueError(f"Unsupported prompt format.")
        regex = r"(I will provide you with) (\d+) (passages)"
        match = re.search(regex, search_text)
        if not match:
            raise ValueError(f"Unsupported prompt format.")
        return int(match.group(2))

    def read_saved_responses(self) -> List[str]:
        num_passages = []
        responses = []
        for dataset in ["dl19", "dl20"]:
            file_name_prefix = f"{self._model_name}_{self._context_size}_{self._top_candidates}_{self._prompt_mode}_{dataset}"
            directory = "prompts_and_responses/BM25"
            for filename in os.listdir(directory):
                if not filename.startswith(file_name_prefix):
                    continue
                file = os.path.join(directory, filename)
                # checking if it is a file
                if not os.path.isfile(file):
                    continue
                with open(file) as f:
                    for line in f:
                        json_obj = json.loads(line)
                        responses.append(json_obj["response"])
                        num_passages.append(self._get_num_passages(json_obj["prompt"]))
        return responses, num_passages

    def _validate_format(self, response: str) -> bool:
        for c in response:
            if not c.isdigit() and c != "[" and c != "]" and c != ">" and c != " ":
                return False
        return True

    def count_errors(
        self, responses: List[str], num_passages: List[int]
    ) -> Dict[str, int]:
        stats_dict = {
            "ok": 0,
            "wrong_format": 0,
            "repetition": 0,
            "missing_documents": 0,
        }
        for resp, num_passage in zip(responses, num_passages):
            if not self._validate_format(resp):
                stats_dict["wrong_format"] += 1
                continue
            begin, end = 0, 0
            raw_resp = resp
            while not resp[begin].isdigit():
                begin += 1
            while not resp[len(resp) - end - 1].isdigit():
                end += 1
            resp = resp[begin : len(resp) - end]
            ranks = resp.split("] > [")
            try:
                ranks = [int(rank) for rank in ranks]
            except ValueError:
                stats_dict["wrong_format"] += 1
                continue
            if len(ranks) < num_passage:
                stats_dict["missing_documents"] += 1
                continue
            if len(ranks) > num_passage or len(set(ranks)) < num_passage:
                stats_dict["repetition"] += 1
                continue
            stats_dict["ok"] += 1
        return stats_dict


def main(args):
    model_name = args.model_name
    context_size = args.context_size
    response_analyzer = ResponseAnalyzer(
        model_name, context_size, 100, PromptMode.RANK_GPT
    )
    responses, num_passages = response_analyzer.read_saved_responses()
    print(response_analyzer.count_errors(responses, num_passages))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--context_size",
        type=int,
        default=4096,
        help="context size used for model",
        required=False,
    )
    args = parser.parse_args()
    main(args)

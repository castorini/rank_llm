import argparse
import json
import os
from typing import List, Dict

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from rank_llm.rankllm import PromptMode


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

    def read_saved_responses(self) -> List[str]:
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
        return responses

    def _validate_format(self, response: str) -> bool:
        for c in response:
            if not c.isdigit() and c != "[" and c != "]" and c != ">" and c != " ":
                return False
        return True

    def count_errors(self, responses: List[str]) -> Dict[str, int]:
        stats_dict = {
            "ok": 0,
            "wrong_format": 0,
            "repetition": 0,
            "missing_documents": 0,
        }
        for resp in responses:
            if not self._validate_format(resp):
                stats_dict["wrong_format"] += 1
                continue
            resp = resp[1:-1]
            ranks = resp.split("] > [")
            ranks = [int(rank) for rank in ranks]
            if len(ranks) < 20:
                stats_dict["missing_documents"] += 1
                continue
            if len(ranks) > 20 or len(set(ranks)) < 20:
                stats_dict["repetition"] += 1
                continue
            stats_dict["ok"] += 1
        return stats_dict


def main(args):
    model_name = args.model_name
    response_analyzer = ResponseAnalyzer(model_name, 4096, 100, PromptMode.RANK_GPT)
    responses = response_analyzer.read_saved_responses()
    print(response_analyzer.count_errors(responses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    args = parser.parse_args()
    main(args)

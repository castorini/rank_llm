import argparse
import json
import os
from typing import List, Dict

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
        files: List[str],
        top_candidates: int,
        prompt_mode: PromptMode,
    ) -> None:
        self._files = files

    def read_saved_responses(self) -> List[str]:
        num_passages = []
        responses = []
        for filename in self._files:
            with open(filename) as f:
                for line in f:
                    json_obj = json.loads(line)
                    responses.append(json_obj["response"])
                    num_passage = filename.split("window_")[1].replace(".json", "")
                    num_passages.append(int(num_passage))
        return responses, num_passages

    def _validate_format(self, response: str) -> bool:
        for c in response:
            if not c.isdigit() and c != "[" and c != "]" and c != ">" and c != " ":
                return False
        return True

    def count_errors(
        self, responses: List[str], num_passages: List[int], verbose: bool = False
    ) -> Dict[str, int]:
        stats_dict = {
            "ok": 0,
            "wrong_format": 0,
            "repetition": 0,
            "missing_documents": 0,
        }
        for resp, num_passage in zip(responses, num_passages):
            if not self._validate_format(resp):
                if verbose:
                    print(resp)
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
                if verbose:
                    print(resp)
                stats_dict["wrong_format"] += 1
                continue
            if len(ranks) < num_passage:
                stats_dict["missing_documents"] += 1
                continue
            if len(ranks) > num_passage or len(set(ranks)) < num_passage:
                stats_dict["repetition"] += 1
                continue
            stats_dict["ok"] += 1
        # Create normalized dicts
        normalized_stats_dict = {}
        for key in stats_dict:
            normalized_stats_dict[key] = (stats_dict[key] / len(responses)) * 100.0
            # Round to two decimal places
            normalized_stats_dict[key] = round(normalized_stats_dict[key], 2)
        return normalized_stats_dict


def main(args):
    response_analyzer = ResponseAnalyzer(args.files, 100, PromptMode.RANK_GPT)
    responses, num_passages = response_analyzer.read_saved_responses()
    print("Normalized scores:")

    print(response_analyzer.count_errors(responses, num_passages, args.verbose))
    # Print normalized scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)

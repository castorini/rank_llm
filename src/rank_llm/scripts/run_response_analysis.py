import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.analysis.response_analysis import ResponseAnalyzer
from rank_llm.rerank.rankllm import PromptMode


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

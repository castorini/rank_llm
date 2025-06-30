import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import Result


class ResponseAnalyzer:
    def __init__(
        self,
        data: Union[List[str], List[Result]],
        use_alpha: bool = False,
    ) -> None:
        self._data = data
        self._use_alpha = use_alpha

    @staticmethod
    def from_inline_results(
        results: List[Result],
        use_alpha: bool = False,
    ) -> "ResponseAnalyzer":
        """
        Method to create a ResponseAnalyzer instance from a list of Result objects.

        Args:
            results (List[Result]): A list of Result objects.
            use_alpha (bool): Whether to evaluate the alphabetical list instead of the numerical one, defaults to False.
            prompt_mode (PromptMode): The prompt mode to use for analysis, defaults to RANK_GPT.

        Returns:
            ResponseAnalyzer: An instance of the ResponseAnalyzer.
        """
        return ResponseAnalyzer(data=results, use_alpha=use_alpha)

    @staticmethod
    def from_stored_files(
        filenames: List[str],
        use_alpha: bool = False,
    ) -> "ResponseAnalyzer":
        """
        Method to create to create a ResponseAnalyzer instance from a list of filenames.

        Args:
            filenames (List[str]): A list of filenames where each file contains data to be analyzed.
            use_alpha (bool): Whether to evaluate the alphabetical list instead of the numerical one, defaults to False.
            prompt_mode (PromptMode): The prompt mode to use for analysis, defaults to RANK_GPT.

        Returns:
            ResponseAnalyzer: An instance of the ResponseAnalyzer.
        """
        return ResponseAnalyzer(data=filenames, use_alpha=use_alpha)

    def read_results_responses(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Reads responses from the specified list of Result objects and produces the total number of passages.

        Returns:
            Tuple[List[str], List[int]]: A tuple object containing a list of responses and a list of corresponding numbers of passages.
        """
        num_passages = []
        responses = []
        output_patterns = self._data[0].invocations_history[0].output_patterns
        for result in self._data:
            for inference_invocation in result.invocations_history:
                responses.append(inference_invocation.response)
                num_passage = self._get_num_passages(
                    inference_invocation.prompt, output_patterns[1]
                )
                num_passages.append(int(num_passage))
        return responses, num_passages, output_patterns

    def read_saved_responses(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Reads responses from the specified list of files and produces the total number of passages.

        Returns:
            Tuple[List[str], List[int]]: A tuple object containing a list of responses and a list of corresponding numbers of passages.
        """
        num_passages = []
        responses = []
        with open(self._data[0]) as f:
            invocations_histories = json.load(f)
            output_patterns = invocations_histories[0]["invocations_history"][0][
                "output_patterns"
            ]
        for result in self._data:
            with open(result) as f:
                invocations_histories = json.load(f)
            for entry in invocations_histories:
                for inference_invocation in entry["invocations_history"]:
                    responses.append(inference_invocation["response"])
                    num_passage = self._get_num_passages(
                        inference_invocation["prompt"], output_patterns[1]
                    )
                    num_passages.append(int(num_passage))
        return responses, num_passages, output_patterns

    def read_responses(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Selects what read response class method to call depending on the input type.

        Returns:
            Tuple[List[str], List[int]]: A tuple object containing a list of responses and a list of corresponding numbers of passages.
        """
        if all(isinstance(item, str) for item in self._data):
            return self.read_saved_responses()
        elif all(isinstance(item, Result) for item in self._data):
            return self.read_results_responses()
        else:
            raise ValueError(
                "Input data must be a list of file paths or a list of Result objects."
            )

    def _validate_format(self, response: str, output_pattern: str) -> bool:
        return bool(re.fullmatch(output_pattern, response.strip()))

    def _get_num_passages(self, prompt, rank_id_pattern: str) -> int:
        search_text = ""
        if isinstance(prompt, str):
            search_text = prompt
        elif isinstance(prompt, list):
            if isinstance(
                prompt[0], dict
            ):  # check if prompt is a list of dicts (e.g., RankGPT style prompts)
                search_text = " ".join([msg["content"] for msg in prompt])
            elif isinstance(
                prompt[0], str
            ):  # check if prompt is a list of strings (e.g., LiT5 style prompts)
                if "text" in prompt[0]:
                    return len(prompt)
                else:
                    raise ValueError(
                        "Unsupported prompt format: for list of strings (RankFID method), each string should be a dict with 'text' key."
                    )
            else:
                raise ValueError("Unsupported prompt format: list of mixed types.")
        else:
            raise ValueError("Unsupported prompt format.")

        matches = re.findall(rank_id_pattern, search_text)
        if not matches:
            raise ValueError(
                "No passage identifiers found in prompt, please fix the prompt template."
            )

        # Use a set to ensure unique passage identifiers incase of examples with duplicate IDs
        return len(set(matches))

    def _process_response(
        self,
        response: str,
        num_passage: int,
        verbose: bool,
        stats_dict: Dict[str, int],
        output_patterns: List[str],
        is_alphabetical: bool = False,
    ):
        resp = response.strip()
        if "</think>" in resp:
            parts = resp.split("</think>")
            resp = parts[-1]
        if len(output_patterns) != 2:
            raise ValueError(
                "Output patterns should contain exactly two elements: one for the expected output format and one for the rank ID pattern."
            )
        output_pattern = output_patterns[0]
        rank_id_pattern = output_patterns[1]

        if not self._validate_format(resp, output_pattern):
            if verbose:
                print(resp)
            stats_dict["wrong_format"] += 1
            return
        matches = re.findall(rank_id_pattern, resp)
        if is_alphabetical:
            ranks = [ord(rank) - ord("A") for rank in matches]
        else:
            ranks = [int(num) for num in matches]
        if not ranks:
            if verbose:
                print(resp)
            stats_dict["wrong_format"] += 1
            return
        if len(ranks) < num_passage:
            stats_dict["missing_documents"] += 1
            return
        if len(ranks) > num_passage or len(set(ranks)) < num_passage:
            stats_dict["repetition"] += 1
            return
        for i in range(num_passage):
            rank = i if is_alphabetical else i + 1
            if rank not in set(ranks):
                stats_dict["missing_documents"] += 1
                return
        stats_dict["ok"] += 1

    def count_errors(
        self, verbose: bool = False, normalize: bool = False
    ) -> Dict[str, Union[int, float]]:
        """
        Counts an array of different types of errors in the given responses.

        Args:
        verbose (bool, optional): When enabled, the analyzer will print out the malformed responses. Defaults to False.
        normalize (bool, optional): When enabled, the returned dictionary will be normalized. Defaults to False.

        Returns:
            Dict[str, Union[int, float]]: A dictionary object containing (normalized) counts of different types of errors.
        """
        responses, num_passages, output_patterns = self.read_responses()

        stats_dict = {
            "ok": 0,
            "wrong_format": 0,
            "repetition": 0,
            "missing_documents": 0,
        }
        for resp, num_passage in zip(responses, num_passages):
            self._process_response(
                response=resp,
                num_passage=num_passage,
                verbose=verbose,
                stats_dict=stats_dict,
                output_patterns=output_patterns,
                is_alphabetical=self._use_alpha,
            )
        if not normalize:
            return stats_dict

        # Create normalized dicts
        normalized_stats_dict = {}
        for key in stats_dict:
            normalized_stats_dict[key] = (stats_dict[key] / len(responses)) * 100.0
            # Round to two decimal places
            normalized_stats_dict[key] = round(normalized_stats_dict[key], 2)
        return normalized_stats_dict


def main(args):
    if args.files:
        response_analyzer = ResponseAnalyzer.from_stored_files(
            args.files, use_alpha=args.use_alpha, prompt_mode=args.prompt_mode
        )
    else:
        print("Error: Please specify the files containing ranking summaries.")
        sys.exit(1)

    error_counts = response_analyzer.count_errors(
        verbose=args.verbose, normalize=args.normalize
    )
    print("Normalized scores:", error_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="+", help="Filenames of ranking summaries", required=False
    )
    parser.add_argument(
        "--use-alpha",
        action="store_true",
        help="Use alphabetical identifiers instead of the numerical ids",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output of errors"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the output dictionary of errors",
    )
    args = parser.parse_args()

    main(args)

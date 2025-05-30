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
from rank_llm.rerank import PromptMode


class ResponseAnalyzer:
    def __init__(
        self,
        data: Union[List[str], List[Result]],
        use_alpha: bool = False,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
    ) -> None:
        self._data = data
        self._use_alpha = use_alpha
        self._prompt_mode = prompt_mode

    @staticmethod
    def from_inline_results(
        results: List[Result],
        use_alpha: bool = False,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
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
        return ResponseAnalyzer(
            data=results, use_alpha=use_alpha, prompt_mode=prompt_mode
        )

    @staticmethod
    def from_stored_files(
        filenames: List[str],
        use_alpha: bool = False,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
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
        return ResponseAnalyzer(
            data=filenames, use_alpha=use_alpha, prompt_mode=prompt_mode
        )

    def read_results_responses(self) -> Tuple[List[str], List[int]]:
        """
        Reads responses from the specified list of Result objects and produces the total number of passages.

        Returns:
            Tuple[List[str], List[int]]: A tuple object containing a list of responses and a list of corresponding numbers of passages.
        """
        num_passages = []
        responses = []
        for result in self._data:
            for inference_invocation in result.invocations_history:
                responses.append(inference_invocation.response)
                num_passage = self._get_num_passages(inference_invocation.prompt)
                num_passages.append(int(num_passage))
        return responses, num_passages

    def read_saved_responses(self) -> Tuple[List[str], List[int]]:
        """
        Reads responses from the specified list of files and produces the total number of passages.

        Returns:
            Tuple[List[str], List[int]]: A tuple object containing a list of responses and a list of corresponding numbers of passages.
        """
        num_passages = []
        responses = []
        for result in self._data:
            with open(result) as f:
                invocations_histories = json.load(f)
            for entry in invocations_histories:
                for inference_invocation in entry["invocations_history"]:
                    responses.append(inference_invocation["response"])
                    num_passage = self._get_num_passages(inference_invocation["prompt"])
                    num_passages.append(int(num_passage))
        return responses, num_passages

    def read_responses(self) -> Tuple[List[str], List[int]]:
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

    def _validate_format(self, response: str) -> bool:
        if self._use_alpha:
            for c in response:
                if not c.isupper() and c not in "[]> ":
                    return False
            return True

        for c in response:
            if not c.isdigit() and c not in "[]> ,":
                return False
        return True

    def _get_num_passages(self, prompt) -> int:
        match self._prompt_mode:
            case PromptMode.LRL:
                assert isinstance(prompt, list)
                assert len(prompt) == 1
                search_text = prompt[0]["content"]
                # Look for PASSAGES=[...] and count the number of passages in the list
                begin = search_text.find("PASSAGES = [")
                search_text = search_text[begin:]
                end = search_text.find("]")
                search_text = search_text[:end]
                return len(search_text.split(", "))
            case PromptMode.LiT5:
                assert type(prompt) == list
                if not prompt:
                    return 0
                # For LiT5, there is one dict with "text" key per passage.
                assert "text" in prompt[0]
                return len(prompt)
            case PromptMode.RANK_GPT:
                search_text = ""
                if type(prompt) == str:
                    search_text = prompt
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
            case PromptMode.RANK_GPT_APEER:
                assert isinstance(prompt, list)
                search_text = ""
                for entry in prompt:
                    search_text += entry["content"]
                # No mention of the total number of passages.
                # Find the last passage identifier instead.
                matches = re.findall(r"\[\d+\]", search_text)
                return int(matches[-1][1:-1])
            case _:
                raise ValueError(f"Unsupported prompt format.")

    def _process_numerical_format(
        self, response: str, num_passage: int, verbose: bool, stats_dict: Dict[str, int]
    ):
        resp = response
        if "</think>" in resp:
            parts = resp.split("</think>")
            resp = parts[-1]
        resp = resp.replace("[rankstart]", "")
        resp = resp.replace("[rankend]", "")
        resp = resp.replace("SORTED_PASSAGES =", "")
        resp = resp.replace(" ", "")
        resp = resp.replace("PASSAGE", "")
        resp = resp.replace("[", "")
        resp = resp.replace("]", "")
        resp = resp.strip()
        if not self._validate_format(resp):
            if verbose:
                print(resp)
            stats_dict["wrong_format"] += 1
            return
        try:
            delim = "," if self._prompt_mode == PromptMode.LRL else ">"
            ranks = resp.split(delim)
            ranks = [int(rank) for rank in ranks]
        except ValueError:
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
            if not i + 1 in set(ranks):
                stats_dict["missing_documents"] += 1
                return
        stats_dict["ok"] += 1

    def _process_alphabetical_format(
        self, response: str, num_passage: int, verbose: bool, stats_dict: Dict[str, int]
    ):
        resp = response.strip()
        if "</think>" in resp:
            parts = resp.split("</think>")
            resp = parts[-1]

        if not self._validate_format(resp):
            if verbose:
                print(resp)
            stats_dict["wrong_format"] += 1
            return
        begin, end = 0, 0
        while begin < len(resp) and not resp[begin].isupper():
            begin += 1
        while end < len(resp) and not resp[len(resp) - end - 1].isupper():
            end += 1
        try:
            resp = resp[begin : len(resp) - end]
            ranks = resp.split("]>[")
            ranks = [ord(rank) - ord("A") for rank in ranks]
        except ValueError:
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
            if not i in set(ranks):
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
        responses, num_passages = self.read_responses()

        stats_dict = {
            "ok": 0,
            "wrong_format": 0,
            "repetition": 0,
            "missing_documents": 0,
        }
        for resp, num_passage in zip(responses, num_passages):
            if self._use_alpha:
                self._process_alphabetical_format(
                    response=resp,
                    num_passage=num_passage,
                    verbose=verbose,
                    stats_dict=stats_dict,
                )
            else:
                self._process_numerical_format(
                    response=resp,
                    num_passage=num_passage,
                    verbose=verbose,
                    stats_dict=stats_dict,
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
        "--prompt-mode",
        type=PromptMode,
        default=PromptMode.RANK_GPT,
        choices=list(PromptMode),
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

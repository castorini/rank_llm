import json
import re
import sys
from typing import Dict, List, Tuple, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.result import Result


# For loading in list of Results
def load_json_input_analyzer(input_json: str) -> Union[List[str], List[Result]]:
    """
    Load input from a JSON file. It is assumed that the a list of dict objects are contained to transform to list of Result objects.
    """
    with open(input_json, "r") as f:
        data = json.load(f)

    # Assuming data is a list of Result objects
    return [Result(**item) for item in data]


class ResponseAnalyzer:
    def __init__(
        self,
        data: Union[List[str], List[Result]],
    ) -> None:
        self._data = data

    def read_results_responses(self) -> Tuple[List[str], List[int]]:
        """
        Reads responses from the specified list of Result objects and produces the total number of passages.

        Returns:
            Tuple[List[str], List[int]]: A tuple object containing a list of responses and a list of corresponding numbers of passages.
        """
        num_passages = []
        responses = []
        for result in self._data:
            for exec_info in result.ranking_exec_summary:
                responses.append(exec_info.response)
                num_passage = self._get_num_passages(exec_info.prompt)
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
                ranking_exec_summaries = json.load(f)
            for summary in ranking_exec_summaries:
                for exec_info in summary["ranking_exec_summary"]:
                    responses.append(exec_info["response"])
                    num_passage = self._get_num_passages(exec_info["prompt"])
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
        for c in response:
            if not c.isdigit() and c != "[" and c != "]" and c != ">" and c != " ":
                return False
        return True

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

    def count_errors(
        self, responses: List[str], num_passages: List[int], verbose: bool = False
    ) -> Dict[str, int]:
        """
        Counts an array of different types of errors in the given responses.

        Args:
            responses (List[str]): A list of response strings.
            num_passages (List[int]): A list of the expected number of passages in each response.
            verbose (bool, optional): If True, prints the erroneous responses. Defaults to False.

        Returns:
            Dict[str, int]: A dictionary object containing counts of different types of errors.
        """
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
    if args.files:
        input_data = args.files
    elif args.input_json_analyzer:
        input_data = load_json_input_analyzer(args.input_json_analyzer)
    else:
        raise ValueError("Either --files or --input_json must be provided.")

    response_analyzer = ResponseAnalyzer(input_data)
    responses, num_passages = response_analyzer.read_responses()

    # Print normalized scores
    print("Normalized scores:")
    print(response_analyzer.count_errors(responses, num_passages, args.verbose))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", required=False)
    parser.add_argument(
        "--input_json_analyzer",
        type=str,
        help="Path to a JSON file containing serialized Result objects.",
        required=False,
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.files and not args.input_json_analyzer:
        parser.error(
            "Either --files or --input_json_analyzer must be provided as arguments."
        )
    main(args)

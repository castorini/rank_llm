import json
from typing import List, Dict, Any


class RankingExecInfo:
    def __init__(
        self, prompt, response: str, input_token_count: int, output_token_count: int
    ):
        self.prompt = prompt
        self.response = response
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count

    def __repr__(self):
        return str(self.__dict__)


class Result:
    def __init__(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        ranking_exec_summary: List[RankingExecInfo] = None,
    ):
        self.query = query
        self.hits = hits
        self.ranking_exec_summary = ranking_exec_summary

    def __repr__(self):
        return str(self.__dict__)


class ResultsWriter:
    def __init__(self, results: List[Result], append: bool = False):
        self._results = results
        self._append = append

    def write_ranking_exec_summary(self, filename: str):
        exec_summary = []
        for result in self._results:
            values = []
            for info in result.ranking_exec_summary:
                values.append(info.__dict__)
            exec_summary.append({"query": result.query, "ranking_exec_summary": values})
        with open(filename, "a" if self._append else "w") as f:
            json.dump(exec_summary, f, indent=2)

    def write_in_json_format(self, filename: str):
        results = []
        for result in self._results:
            results.append({"query": result.query, "hits": result.hits})
        with open(filename, "a" if self._append else "w") as f:
            json.dump(results, f, indent=2)

    def write_in_trec_eval_format(self, filename: str):
        with open(filename, "a" if self._append else "w") as f:
            for result in self._results:
                for hit in result.hits:
                    f.write(
                        f"{hit['qid']} Q0 {hit['docid']} {hit['rank']} {hit['score']} rank\n"
                    )

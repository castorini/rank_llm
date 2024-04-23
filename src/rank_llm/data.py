import json
from typing import List, TypedDict
from typing_extensions import NotRequired


class Query:
    def __init__(self, text: str, qid: str = None):
        self.text = text
        self.qid = qid

    def __repr__(self):
        return str(self.__dict__)


class Candidate:
    def __init__(self, docid: str, score: str, content: str, title: str = None):
        self.docid = docid
        self.score = score
        self.content = content
        self.title = title

    def __repr__(self):
        return str(self.__dict__)


class Request:
    def __init__(
        self,
        query: Query,
        candidates: List[Candidate],
    ):
        self.query = query
        self.candidates = candidates

    def __repr__(self):
        return str(self.__dict__)


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
        query: Query,
        candidates: List[Candidate],
        ranking_exec_summary: List[RankingExecInfo] = None,
    ):
        self.query = query
        self.candidates: candidates
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
            results.append({"query": result.query, "candidates": result.candidates})
        with open(filename, "a" if self._append else "w") as f:
            json.dump(results, f, indent=2)

    def write_in_jsonl_format(self, filename: str):
        with open(filename, "a" if self._append else "w") as f:
            for result in self._results:
                json.dump({"query": result.query, "candidates": result.candidates}, f)
                f.write("\n")

    def write_in_trec_eval_format(self, filename: str):
        with open(filename, "a" if self._append else "w") as f:
            for result in self._results:
                qid = result.query.qid
                for rank, cand in enumerate(result.candidates, start=1):
                    f.write(f"{qid} Q0 {cand['docid']} {rank} {cand['score']} rank\n")

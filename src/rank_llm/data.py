import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Union

from dacite import from_dict


@dataclass
class Query:
    text: str
    qid: Union[str | int]


@dataclass
class Candidate:
    docid: Union[str | int]
    score: float
    doc: Dict[str, Any]


@dataclass
class Request:
    query: Query
    candidates: List[Candidate] = field(default_factory=list)


@dataclass
class InferenceInvocation:
    prompt: Any
    response: str
    input_token_count: int
    output_token_count: int


@dataclass
class Result:
    query: Query
    candidates: list[Candidate] = field(default_factory=list)
    invocations_history: list[InferenceInvocation] = (field(default_factory=list),)


@dataclass
class TemplateSectionConfig:
    required: bool
    required_placeholders: Set[str]
    allowed_placeholders: Set[str]


def read_requests_from_file(file_path: str) -> List[Request]:
    extension = file_path.split(".")[-1]
    if extension == "jsonl":
        requests = []
        with open(file_path, "r") as f:
            for l in f:
                if not l.strip():
                    continue
                requests.append(from_dict(data_class=Request, data=json.loads(l)))
        return requests
    elif extension == "json":
        with open(file_path, "r") as f:
            request_dicts = json.load(f)
        return [
            from_dict(data_class=Request, data=request_dict)
            for request_dict in request_dicts
        ]
    else:
        raise ValueError(f"Expected json or jsonl file format, got {extension}")


class DataWriter:
    def __init__(
        self,
        data: Union[Request | Result | List[Result] | List[Request]],
        append: bool = False,
    ):
        if isinstance(data, list):
            self._data = data
        else:
            self._data = [data]
        self._append = append

    def write_inference_invocations_history(self, filename: str):
        aggregated_history = []
        for d in self._data:
            values = []
            for info in d.invocations_history:
                values.append(info.__dict__)
            aggregated_history.append(
                {"query": d.query.__dict__, "invocations_history": values}
            )
        with open(filename, "a" if self._append else "w") as f:
            output = json.dumps(aggregated_history, indent=2, ensure_ascii=False)
            f.write(output)

    def write_in_json_format(self, filename: str):
        results = []
        for d in self._data:
            candidates = [candidate.__dict__ for candidate in d.candidates]
            results.append({"query": d.query.__dict__, "candidates": candidates})
        with open(filename, "a" if self._append else "w") as f:
            output = json.dumps(results, indent=2, ensure_ascii=False)
            f.write(output)

    def write_in_jsonl_format(self, filename: str):
        with open(filename, "a" if self._append else "w") as f:
            for d in self._data:
                candidates = [candidate.__dict__ for candidate in d.candidates]
                output = json.dumps(
                    {"query": d.query.__dict__, "candidates": candidates},
                    ensure_ascii=False,
                )
                f.write(output)
                f.write("\n")

    def write_in_trec_eval_format(self, filename: str):
        with open(filename, "a" if self._append else "w") as f:
            for d in self._data:
                qid = d.query.qid
                for rank, cand in enumerate(d.candidates, start=1):
                    f.write(f"{qid} Q0 {cand.docid} {rank} {cand.score} rank_llm\n")

import json
from dataclasses import dataclass, field
from typing import Any

from dacite import from_dict


@dataclass
class Query:
    text: str
    qid: str | int


@dataclass
class Candidate:
    docid: str | int
    score: float
    doc: dict[str, Any]


@dataclass
class Request:
    query: Query
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class InferenceInvocation:
    prompt: Any
    response: str
    input_token_count: int
    output_token_count: int
    reasoning: str | None = None
    token_usage: dict[str, Any] | None = None
    output_validation_regex: str | None = None
    output_extraction_regex: str | None = None


@dataclass
class Result:
    query: Query
    candidates: list[Candidate] = field(default_factory=list)
    invocations_history: list[InferenceInvocation] = (field(default_factory=list),)


@dataclass
class TemplateSectionConfig:
    required: bool
    required_placeholders: set[str]
    allowed_placeholders: set[str]


def read_requests_from_file(file_path: str) -> list[Request]:
    extension = file_path.split(".")[-1]
    if extension == "jsonl":
        requests = []
        with open(file_path) as f:
            for line in f:
                if not line.strip():
                    continue
                requests.append(from_dict(data_class=Request, data=json.loads(line)))
        return requests
    elif extension == "json":
        with open(file_path) as f:
            request_dicts = json.load(f)
        return [
            from_dict(data_class=Request, data=request_dict)
            for request_dict in request_dicts
        ]
    else:
        raise ValueError(f"Expected json or jsonl file format, got {extension}")


class DataWriter:
    """Write results to a file, merging with an existing JSON array when appending."""

    def __init__(
        self,
        data: Request | Result | list[Result] | list[Request],
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
        self._write_json_array(filename, aggregated_history)

    def _write_json_array(self, filename: str, records: list[dict[str, Any]]):
        if self._append:
            try:
                with open(filename) as f:
                    contents = f.read()
                    existing = json.loads(contents) if contents.strip() else []
            except FileNotFoundError:
                existing = []
            if not isinstance(existing, list):
                raise ValueError(f"Expected a JSON array in {filename}")
            records = existing + records
        with open(filename, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def write_in_json_format(self, filename: str):
        results = []
        for d in self._data:
            candidates = [candidate.__dict__ for candidate in d.candidates]
            results.append({"query": d.query.__dict__, "candidates": candidates})
        self._write_json_array(filename, results)

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

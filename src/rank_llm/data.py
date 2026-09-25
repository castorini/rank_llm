import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    invocations_history: list[InferenceInvocation] = field(default_factory=list)


@dataclass
class TemplateSectionConfig:
    required: bool
    required_placeholders: set[str]
    allowed_placeholders: set[str]


class RerankValidationError(ValueError):
    """Invalid user input, as distinct from an execution failure."""


def normalize_rerank_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate query/candidate forms and supply missing IDs and scores."""
    if (
        not isinstance(payload, dict)
        or "query" not in payload
        or "candidates" not in payload
    ):
        raise RerankValidationError("payload must contain query and candidates")
    if any(payload.get(key) for key in ("dataset", "requests_file", "retriever_host")):
        raise RerankValidationError(
            "direct candidates cannot be combined with retrieval sources"
        )
    query = payload["query"]
    if isinstance(query, str):
        query_text, query_id = query, ""
    elif isinstance(query, dict) and isinstance(query.get("text"), str):
        query_text, query_id = query["text"], query.get("qid", "")
    else:
        raise RerankValidationError(
            "query must be a string or an object containing text"
        )
    if type(query_id) not in (str, int):
        raise RerankValidationError("query ID must be a string or integer")
    if not isinstance(payload["candidates"], list):
        raise RerankValidationError("candidates must be an array")
    candidates = []
    for index, candidate in enumerate(payload["candidates"], start=1):
        if isinstance(candidate, str):
            candidate = {"doc": candidate}
        if not isinstance(candidate, dict) or not (
            "text" in candidate or "doc" in candidate
        ):
            raise RerankValidationError(f"candidate {index} must contain text or doc")
        if "text" in candidate and not isinstance(candidate["text"], str):
            raise RerankValidationError(f"candidate {index} text must be a string")
        doc = candidate.get("text", candidate.get("doc"))
        if not isinstance(doc, str | dict):
            raise RerankValidationError(
                f"candidate {index} document must be a string or object"
            )
        docid = candidate.get("docid", str(index))
        score = candidate.get("score", 0.0)
        if type(docid) not in (str, int):
            raise RerankValidationError(
                f"candidate {index} docid must be a string or integer"
            )
        if type(score) not in (int, float) or not math.isfinite(score):
            raise RerankValidationError(
                f"candidate {index} score must be a finite number"
            )
        candidates.append(
            {
                "docid": docid,
                "score": score,
                "doc": {"contents": doc} if isinstance(doc, str) else doc,
            }
        )
    return {"query_text": query_text, "query_id": query_id, "candidates": candidates}


def read_requests_from_file(file_path: str) -> list[Request]:
    """Read JSON/JSONL requests, accepting structured documents or text candidates."""
    path = Path(file_path)
    if not path.is_file():
        raise RerankValidationError(f"missing file: {file_path}")

    def parse_record(payload: Any) -> Request:
        if isinstance(payload, dict) and isinstance(payload.get("query"), dict):
            # Request.candidates has always defaulted to an empty list in files.
            payload = {"candidates": [], **payload}
        record = normalize_rerank_input(payload)
        return Request(
            query=Query(text=record["query_text"], qid=record["query_id"]),
            candidates=[Candidate(**c) for c in record["candidates"]],
        )

    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".json":
            try:
                payloads = json.load(handle)
            except json.JSONDecodeError as exc:
                raise RerankValidationError(f"invalid JSON: {exc.msg}") from exc
            if not isinstance(payloads, list):
                raise RerankValidationError("request JSON file must contain an array")
            return [parse_record(payload) for payload in payloads]
        if path.suffix == ".jsonl":
            requests = []
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    requests.append(parse_record(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise RerankValidationError(
                        f"invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc
                except RerankValidationError as exc:
                    raise RerankValidationError(
                        f"invalid request on line {line_number}: {exc}"
                    ) from exc
            return requests
        raise RerankValidationError("request file must use .json or .jsonl")


class DataWriter:
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

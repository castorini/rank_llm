"""Canonical query sets for named datasets, independent of retrieval indexes."""

from rank_llm._optional import install_hint
from rank_llm.data import Request
from rank_llm.retrieve.topics_dict import TOPICS


class CachedQueryValidationError(ValueError):
    """A readable retrieval cache does not match the requested dataset."""


def load_dataset_topics(dataset: str) -> tuple[dict, dict]:
    """Load topics and qrels using the same mapping as live retrieval.

    Live retrieval includes only topics with qrels. Qrels membership, rather
    than positive relevance, defines this benchmark subset.
    """
    try:
        from pyserini.search import get_qrels, get_topics
    except ImportError as err:
        raise ImportError(install_hint("pyserini")) from err

    if dataset in TOPICS:
        topics_key = dataset if dataset in {"dl20", "dl21", "dl22", "dl23"} else TOPICS[dataset]
        return get_topics(topics_key), get_qrels(TOPICS[dataset])

    if dataset.startswith("beir-v1.0.0-") and dataset.endswith((".flat", ".multifield")):
        dataset = dataset.removesuffix(".flat").removesuffix(".multifield")
    try:
        return get_topics(dataset), get_qrels(dataset)
    except Exception as err:
        raise ValueError(f"Invalid collection name: {dataset} - {err}") from err


def validate_cached_queries(
    requests: list[Request], topics: dict, qrels: dict, dataset: str, source: str
) -> list[Request]:
    """Validate and order requests by canonical topics.

    Query IDs are compared as strings (Pyserini may use integer IDs). Query
    text must match exactly, including whitespace, case, and punctuation.
    """
    judged_ids = {str(qid) for qid in qrels}
    expected = {str(qid): topic["title"] for qid, topic in topics.items() if str(qid) in judged_ids}
    by_id = {}
    duplicates = set()
    mismatched = set()
    for request in requests:
        qid = str(request.query.qid)
        if qid in by_id:
            duplicates.add(qid)
        by_id[qid] = request
        if qid in expected and request.query.text != expected[qid]:
            mismatched.add(qid)

    problems = []
    for label, ids in (
        ("missing query IDs", expected.keys() - by_id.keys()),
        ("extra query IDs", by_id.keys() - expected.keys()),
        ("duplicate query IDs", duplicates),
        ("mismatched query text for IDs", mismatched),
    ):
        if ids:
            problems.append(f"{label}: {', '.join(sorted(ids))}")
    if problems:
        raise CachedQueryValidationError(
            f"Invalid retrieval cache {source!r} for dataset {dataset!r}: " + "; ".join(problems)
        )
    return [by_id[qid] for qid in expected]

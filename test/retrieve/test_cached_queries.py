import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from rank_llm.data import Candidate, Query, Request
from rank_llm.retrieve.dataset_topics import (
    CachedQueryValidationError,
    load_dataset_topics,
    validate_cached_queries,
)
from rank_llm.retrieve.retriever import RetrievalMethod, RetrievalMode, Retriever


class TestCachedQueries(unittest.TestCase):
    def setUp(self):
        self.topics = {
            1: {"title": "first query"},
            2: {"title": "second query"},
            3: {"title": "unjudged"},
        }
        self.qrels = {1: {}, 2: {}}
        self.requests = [Request(Query("second query", "2")), Request(Query("first query", 1))]

    def validate(self, requests):
        return validate_cached_queries(requests, self.topics, self.qrels, "dl19", "cache.jsonl")

    def test_valid_cache_uses_canonical_order_and_judged_subset(self):
        self.assertEqual([str(r.query.qid) for r in self.validate(self.requests)], ["1", "2"])

    def test_invalid_queries(self):
        cases = [
            (self.requests[:1], "missing query IDs: 1"),
            (self.requests + [Request(Query("extra", 4))], "extra query IDs: 4"),
            (self.requests + [self.requests[0]], "duplicate query IDs: 2"),
            ([Request(Query("First query", 1)), self.requests[0]], "mismatched query text for IDs: 1"),
            ([], "missing query IDs: 1, 2"),
        ]
        for requests, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(CachedQueryValidationError, message),
            ):
                self.validate(requests)

    def test_query_text_requires_exact_match(self):
        for text in (
            " first query",
            "first query ",
            "first  query",
            "first query\n",
            "First query",
            "first query.",
        ):
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(CachedQueryValidationError, "mismatched query text for IDs: 1"),
            ):
                self.validate([Request(Query(text, 1)), self.requests[0]])

    def test_local_and_downloaded_caches(self):
        # Validation errors must escape both fallback handlers, before any reranking.
        for local in (True, False):
            for valid in (True, False):
                with (
                    self.subTest(local=local, valid=valid),
                    tempfile.TemporaryDirectory(dir="tmp") as directory,
                ):
                    cache = Path(directory) / "cache.jsonl"
                    requests = self.requests if valid else self.requests[:1]
                    for request in requests:
                        request.candidates = [Candidate("a", 1.0, {}), Candidate("b", 0.0, {})]
                    cache.write_text("".join(json.dumps(asdict(r)) + "\n" for r in requests))
                    module = "rank_llm.retrieve.retriever"
                    with (
                        patch(f"{module}.load_dataset_topics", return_value=(self.topics, self.qrels)),
                        patch(f"{module}.get_cache_home", return_value=directory),
                        patch.object(
                            Retriever,
                            "_get_file_with_highest_k",
                            return_value=(str(cache), 100) if local else (None, -1),
                        ),
                        patch(f"{module}.download_cached_hits", return_value=str(cache)) as download,
                    ):
                        retriever = Retriever(RetrievalMode.DATASET, "dl19", RetrievalMethod.BM25)
                        if valid:
                            results = retriever.retrieve(k=1)
                            self.assertEqual([str(r.query.qid) for r in results], ["1", "2"])
                            self.assertTrue(all(len(r.candidates) == 1 for r in results))
                        else:
                            with self.assertRaises(CachedQueryValidationError):
                                retriever.retrieve(k=1)
                        self.assertEqual(download.call_count, 0 if local else 1)

    def test_topic_mapping(self):
        with (
            patch("pyserini.search.get_topics", return_value=self.topics) as topics,
            patch("pyserini.search.get_qrels", return_value=self.qrels) as qrels,
        ):
            for dataset, topic_key, qrels_key in (
                ("dl19", "dl19-passage", "dl19-passage"),
                ("dl20", "dl20", "dl20-passage"),
                ("covid", "beir-v1.0.0-trec-covid-test", "beir-v1.0.0-trec-covid-test"),
                ("beir-v1.0.0-nq-test.flat", "beir-v1.0.0-nq-test", "beir-v1.0.0-nq-test"),
            ):
                self.assertEqual(load_dataset_topics(dataset), (self.topics, self.qrels))
                topics.assert_called_with(topic_key)
                qrels.assert_called_with(qrels_key)

    def test_validation_precedes_max_queries(self):
        from rank_llm.retrieve_and_rerank import retrieve_and_rerank

        with (
            tempfile.TemporaryDirectory(dir="tmp") as directory,
            patch("rank_llm.retrieve_and_rerank.Reranker") as reranker,
            patch("rank_llm.retrieve.retriever.load_dataset_topics", return_value=(self.topics, self.qrels)),
            patch.object(Retriever, "_get_file_with_highest_k") as local_cache,
        ):
            cache = Path(directory) / "cache.jsonl"
            # The first query is valid, but the complete cache is missing query 2.
            cache.write_text(json.dumps(asdict(self.requests[1])) + "\n")
            local_cache.return_value = (str(cache), 100)
            with self.assertRaisesRegex(CachedQueryValidationError, "missing query IDs: 2"):
                retrieve_and_rerank("unused", query="", dataset="dl19", max_queries=1)
            reranker.return_value.rerank_batch.assert_not_called()

    def test_explicit_requests_file_defines_its_own_queries(self):
        from rank_llm.retrieve_and_rerank import retrieve

        with (
            tempfile.TemporaryDirectory(dir="tmp") as directory,
            patch("rank_llm.retrieve.retriever.load_dataset_topics") as topics,
        ):
            cache = Path(directory) / "requests.jsonl"
            cache.write_text(json.dumps(asdict(Request(Query("custom query", "custom")))) + "\n")
            requests = retrieve(retrieval_mode=RetrievalMode.CACHED_FILE, requests_file=str(cache))
            self.assertEqual(requests[0].query, Query("custom query", "custom"))
            topics.assert_not_called()

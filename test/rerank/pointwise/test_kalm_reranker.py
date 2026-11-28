import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rank_llm.cli.main import main
from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank import Reranker
from rank_llm.rerank.pointwise import KaLMReranker

MODEL_ID = "KaLM-Embedding/KaLM-Reranker-V1-Nano-R2"
INSTRUCTION = "Given a query, retrieve documents that answer the query."
MODULE = "rank_llm.rerank.pointwise.kalm_reranker"
LOAD_CLASS = "transformers.dynamic_module_utils.get_class_from_dynamic_module"
SCORER_CLASS = "kalm_reranker.KaLMReranker"


class TestKaLMReranker(unittest.TestCase):
    @patch(f"{MODULE}.snapshot_download")
    @patch(LOAD_CLASS)
    def test_hub_checkpoint_dispatch_forwards_scorer_options(
        self, load_class, snapshot_download
    ):
        snapshot = "/cached/kalm/snapshots/revision"
        snapshot_download.return_value = snapshot

        coordinator = Reranker.create_model_coordinator(
            MODEL_ID,
            None,
            interactive=False,
            device="cpu",
            batch_size=7,
            context_size=256,
            encoder_chunk_size=8,
        )

        self.assertIsInstance(coordinator, KaLMReranker)
        snapshot_download.assert_called_once_with(MODEL_ID)
        load_class.assert_called_once_with(SCORER_CLASS, snapshot)
        load_class.return_value.assert_called_once_with(
            snapshot, device="cpu", batch_size=7, max_length=256, chunk_size=8
        )

    @patch(f"{MODULE}.snapshot_download")
    @patch(LOAD_CLASS)
    def test_local_checkpoint_reranks_with_yaml_and_preserves_scores(
        self, load_class, snapshot_download
    ):
        scorer = load_class.return_value.return_value
        scorer.tokenizer.encode.return_value = [1]
        scorer.predict.return_value = [0.13, 0.87]
        query = "What is the capital of China?"
        unrelated = "Gravity attracts bodies."
        relevant = "Beijing is the capital."
        request = Request(
            Query(text=query, qid="q1"),
            [
                Candidate(docid="low", score=0.8, doc={"text": unrelated}),
                Candidate(docid="high", score=0.4, doc={"text": relevant}),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            # Pinned local snapshots can have hash names without "KaLM".
            checkpoint = Path(directory).resolve() / "012345abcdef"
            checkpoint.mkdir()
            (checkpoint / "kalm_reranker.py").touch()
            coordinator = Reranker.create_model_coordinator(
                str(checkpoint), None, interactive=False, device="cpu"
            )
            result = coordinator.rerank_batch(
                [request], populate_invocations_history=True
            )[0]

        self.assertIsInstance(coordinator, KaLMReranker)
        snapshot_download.assert_not_called()
        load_class.assert_called_once_with(SCORER_CLASS, str(checkpoint))
        load_class.return_value.assert_called_once_with(
            str(checkpoint), device="cpu", batch_size=32, max_length=1024, chunk_size=4
        )
        scorer.predict.assert_called_once_with(
            [(query, unrelated), (query, relevant)], instruction=INSTRUCTION
        )
        self.assertEqual(
            result.invocations_history[0].prompt,
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "query", "content": query},
                {"role": "document", "content": unrelated},
            ],
        )
        self.assertEqual(
            [(candidate.docid, candidate.score) for candidate in result.candidates],
            [("high", 0.87), ("low", 0.13)],
        )

    @patch(f"{MODULE}._load_checkpoint")
    def test_cli_reranks_cached_requests_to_trec(self, load_checkpoint):
        scorer = load_checkpoint.return_value
        scorer.tokenizer.encode.return_value = [1]
        scorer.predict.return_value = [0.42]
        payload = {
            "query": {"text": "question", "qid": "q1"},
            "candidates": [{"docid": "d1", "score": 0.0, "doc": {"text": "passage"}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            requests = Path(directory) / "requests.jsonl"
            requests.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            run = Path(directory) / "run.trec"
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exit_code = main(
                    [
                        "rerank",
                        "--model-path",
                        MODEL_ID,
                        "--requests-file",
                        str(requests),
                        "--output-trec-file",
                        str(run),
                    ]
                )

            self.assertEqual(exit_code, 0, output.getvalue())
            self.assertEqual(run.read_text(), "q1 Q0 d1 1 0.42 rank_llm\n")

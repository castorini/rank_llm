"""Run with FIQA defaults, or reuse a local index and topics file.

Requires the pyserini and vllm extras, JDK 21, and a compatible GPU.
Example bounded run (retrieval still processes all topics):
    python src/rank_llm/demo/rerank_dataset_with_custom_index.py \
        --num-queries 1 --k 4 --window-size 2 --stride 1 --batch-size 1

Existing paths are reused. For missing custom paths, supply --index-url and/or
--topics-url. The archive must contain a directory named like the index path.
Outputs are <output-dir>/<model>/rerank.jsonl, rerank.txt, and invocations.json;
disabling history removes the previous invocations.json in that directory.
"""

import argparse
import os
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.request import urlretrieve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(parent)

from rank_llm.data import DataWriter
from rank_llm.rerank.listwise import VicunaReranker
from rank_llm.retrieve import Retriever

DEFAULT_INDEX_PATH = "indexes/lucene-index.beir-v1.0.0-fiqa.flat.20221116.505594"
DEFAULT_INDEX_URL = "https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.beir-v1.0.0-fiqa.flat.20221116.505594.tar.gz"
DEFAULT_TOPICS_PATH = "topics/topics.beir-v1.0.0-fiqa.test.tsv.gz"
DEFAULT_TOPICS_URL = "https://raw.githubusercontent.com/castorini/eval/0b4acbd929edd11edfd16250457fb70ff69e9b4f/topics/topics.beir-v1.0.0-fiqa.test.tsv.gz"
DEFAULT_MODEL = "castorini/rank_vicuna_7b_v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve and rerank a dataset using a custom Pyserini index."
    )
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument(
        "--index-url",
        help="Download an archive containing a directory matching --index-path's basename. Defaults to the FIQA URL only for the default index path.",
    )
    parser.add_argument("--topics-path", default=DEFAULT_TOPICS_PATH)
    parser.add_argument(
        "--topics-url",
        help="Download missing topics. Defaults to the FIQA URL only for the default topics path.",
    )
    parser.add_argument("--index-type", choices=("lucene", "impact"), default="lucene")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--onnx", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of candidates to retrieve and rerank (default: 100).",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Limit reranking to the first N queries after retrieval (default: all).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output-dir", default="demo_outputs/custom_index")
    parser.add_argument(
        "--populate-invocations-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _prepare_data(args: argparse.Namespace) -> None:
    index_path = Path(args.index_path)
    topics_path = Path(args.topics_path)
    index_url = args.index_url
    topics_url = args.topics_url
    if index_path == Path(DEFAULT_INDEX_PATH) and index_url is None:
        index_url = DEFAULT_INDEX_URL
    if topics_path == Path(DEFAULT_TOPICS_PATH) and topics_url is None:
        topics_url = DEFAULT_TOPICS_URL
    if index_path.exists() and not index_path.is_dir():
        raise ValueError(f"Index path is not a directory: {index_path}")
    if topics_path.exists() and not topics_path.is_file():
        raise ValueError(f"Topics path is not a file: {topics_path}")
    if not index_path.exists() and not index_url:
        raise ValueError(
            "Missing custom index; provide an existing --index-path or --index-url"
        )
    if not topics_path.exists() and not topics_url:
        raise ValueError(
            "Missing custom topics; provide an existing --topics-path or --topics-url"
        )
    if not index_path.exists():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=index_path.parent) as temporary:
            staging = Path(temporary)
            archive_path = staging / "download.tar.gz"
            urlretrieve(index_url, archive_path)
            extracted = staging / "extracted"
            extracted.mkdir()
            with tarfile.open(archive_path) as archive:
                archive.extractall(extracted, filter="data")
            downloaded_index = extracted / index_path.name
            if not downloaded_index.is_dir() or downloaded_index.is_symlink():
                raise ValueError(
                    f"Index archive must contain a directory named {index_path.name}"
                )
            downloaded_index.rename(index_path)
    if not topics_path.exists():
        topics_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=topics_path.parent) as temporary:
            downloaded_topics = Path(temporary) / "topics"
            urlretrieve(topics_url, downloaded_topics)
            downloaded_topics.replace(topics_path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    positive_args = (
        "k",
        "batch_size",
        "context_size",
        "window_size",
        "stride",
        "num_gpus",
    )
    for name in positive_args:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    if args.num_queries is not None and args.num_queries <= 0:
        parser.error("--num-queries must be greater than 0")
    if not 0 < args.stride <= args.window_size:
        parser.error("--stride must be greater than 0 and no larger than --window-size")

    try:
        _prepare_data(args)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    retrieved_results = Retriever.from_custom_index(
        index_path=args.index_path,
        topics_path=args.topics_path,
        index_type=args.index_type,
        encoder=args.encoder,
        onnx=args.onnx,
        k=args.k,
    )
    if args.num_queries is not None:
        retrieved_results = retrieved_results[: args.num_queries]
    if not retrieved_results:
        parser.error("the topics file contains no queries")

    reranker = VicunaReranker(
        model_path=args.model,
        context_size=args.context_size,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
    )
    try:
        rerank_results = reranker.rerank_batch(
            requests=retrieved_results,
            rank_end=args.k,
            top_k_retrieve=args.k,
            populate_invocations_history=args.populate_invocations_history,
        )
    finally:
        reranker.close()

    print(rerank_results)
    model_name = args.model.rsplit("/", 1)[-1].lower()
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = DataWriter(rerank_results)
    writer.write_in_jsonl_format(str(output_dir / "rerank.jsonl"))
    writer.write_in_trec_eval_format(str(output_dir / "rerank.txt"))
    history_path = output_dir / "invocations.json"
    if args.populate_invocations_history:
        writer.write_inference_invocations_history(str(history_path))
    else:
        history_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

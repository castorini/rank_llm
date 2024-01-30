from enum import Enum
import json
from pathlib import Path
from typing import Dict, List

from pyserini.index import IndexReader
from pyserini.search import (
    LuceneSearcher,
    LuceneImpactSearcher,
    FaissSearcher,
    QueryEncoder,
    get_topics,
    get_qrels,
)
from tqdm import tqdm

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.result import Result, ResultsWriter
from rank_llm.retrieve.indices_dict import INDICES
from rank_llm.retrieve.topics_dict import TOPICS


class RetrievalMethod(Enum):
    UNSPECIFIED = "unspecified"
    BM25 = "bm25"
    BM25_RM3 = "bm25_rm3"
    SPLADE_P_P_ENSEMBLE_DISTIL = "SPLADE++_EnsembleDistil_ONNX"
    D_BERT_KD_TASB = "distilbert_tas_b"
    OPEN_AI_ADA2 = "openai-ada2"
    REP_LLAMA = "rep-llama"

    def __str__(self):
        return self.value


class PyseriniRetriever:
    def __init__(
        self, dataset: str, retrieval_method: RetrievalMethod = RetrievalMethod.BM25
    ) -> None:
        self._dataset = dataset
        self._retrieval_method = retrieval_method
        if retrieval_method in [RetrievalMethod.BM25, RetrievalMethod.BM25_RM3]:
            self._searcher = LuceneSearcher.from_prebuilt_index(self._get_index())
            if not self._searcher:
                raise ValueError(
                    f"Could not create searcher for `{dataset}` dataset from prebuilt `{self._get_index()}` index."
                )
            self._searcher.set_bm25()
            if retrieval_method == RetrievalMethod.BM25_RM3:
                self._searcher.set_rm3()
        elif retrieval_method == RetrievalMethod.SPLADE_P_P_ENSEMBLE_DISTIL:
            self._searcher = LuceneImpactSearcher.from_prebuilt_index(
                self._get_index(),
                query_encoder="SpladePlusPlusEnsembleDistil",
                min_idf=0,
                encoder_type="onnx",
            )
            if not self._searcher:
                raise ValueError(
                    f"Could not create impact searcher for `{dataset}` dataset from prebuilt `{self._get_index()}` index."
                )
        elif retrieval_method in [
            RetrievalMethod.D_BERT_KD_TASB,
            RetrievalMethod.OPEN_AI_ADA2,
        ]:
            query_encoders_map = {
                (
                    RetrievalMethod.D_BERT_KD_TASB,
                    "dl19",
                ): "distilbert_tas_b-dl19-passage",
                (RetrievalMethod.D_BERT_KD_TASB, "dl20"): "distilbert_tas_b-dl20",
                (RetrievalMethod.OPEN_AI_ADA2, "dl19"): "openai-ada2-dl19-passage",
                (RetrievalMethod.OPEN_AI_ADA2, "dl20"): "openai-ada2-dl20",
            }
            query_encoder = QueryEncoder.load_encoded_queries(
                query_encoders_map[(retrieval_method, dataset)]
            )
            self._searcher = FaissSearcher.from_prebuilt_index(
                self._get_index(), query_encoder
            )
            if not self._searcher:
                raise ValueError(
                    f"Could not create faiss searcher for `{dataset}` dataset from prebuilt `{self._get_index()}` index."
                )
        else:
            raise ValueError(
                "Unsupported/Invalid retrieval method: %s" % retrieval_method
            )
        if dataset not in TOPICS:
            raise ValueError("dataset %s not in TOPICS" % dataset)
        if dataset in ["dl20", "dl21", "dl22"]:
            topics_key = dataset
        else:
            topics_key = TOPICS[dataset]
        self._topics = get_topics(topics_key)
        self._qrels = get_qrels(TOPICS[dataset])
        self._index_reader = IndexReader.from_prebuilt_index(self._get_index("bm25"))

    def _get_index(self, key: str = None) -> str:
        if not key:
            key = self._retrieval_method.value
            # bm25_rm3 uses the same indices as bm25
            if key == "bm25_rm3":
                key = "bm25"
        if self._dataset not in INDICES[key]:
            raise ValueError("dataset %s not in INDICES[%s]" % self._dataset, key)
        return INDICES[key][self._dataset]

    def _retrieve_query(
        self, query: str, ranks: List[Dict[str, any]], k: int, qid=None
    ) -> None:
        hits = self._searcher.search(query, k=k)
        ranks.append(Result(query=query, hits=[]))
        rank = 0
        for hit in hits:
            rank += 1
            document = self._index_reader.doc(hit.docid)
            content = json.loads(document.raw())
            if "title" in content:
                content = (
                    "Title: " + content["title"] + " " + "Content: " + content["text"]
                )
            elif "contents" in content:
                content = content["contents"]
            else:
                content = content["passage"]
            content = " ".join(content.split())
            # hit.score could be of type 'numpy.float32' which is not json serializable. Always explicitly cast it to float.
            ranks[-1].hits.append(
                {
                    "content": content,
                    "qid": qid,
                    "docid": hit.docid,
                    "rank": rank,
                    "score": float(hit.score),
                }
            )

    def retrieve(self, k=100, qid=None) -> List[Result]:
        """
        Retrieves documents for each query, specified by query id `qid`, in the configured topics.
        Returns list of retrieved documents with specified ranking.

        Args:
            k (int, optional): The number of documents to retrieve for each query. Defaults to 100.
            qid (optional): Specific query ID to retrieve for. Defaults to None.

        Returns:
            List[Result]: A list of retrieval results.
        """
        ranks = []
        if isinstance(self._topics, str):
            self._retrieve_query(self._topics, ranks, k, qid)
            return ranks

        for qid in tqdm(self._topics):
            if qid in self._qrels:
                query = self._topics[qid]["title"]
                self._retrieve_query(query, ranks, k, qid)
        return ranks

    def num_queries(self) -> int:
        """
        Returns the number of queries in the configured topics list.

        Returns:
            int: The number of queries.
        """
        if isinstance(self._topics, str):
            return 1
        return len(self._topics)

    def retrieve_and_store(
        self, k=100, qid=None, store_trec: bool = True, store_qrels: bool = True
    ) -> List[Result]:
        """
        Retrieves documents and stores the results in the given formats.

        Args:
            k (int, optional): The number of documents to retrieve for each query. Defaults to 100.
            qid (optional): Specific query ID to retrieve for. Defaults to None.
            store_trec (bool, optional): Flag to store results in TREC format. Defaults to True.
            store_qrels (bool, optional): Flag to store QRELS of the dataset. Defaults to True.

        Returns:
            List[Result]: The retrieval results.
        """
        results = self.retrieve(k, qid)
        Path("retrieve_results/").mkdir(parents=True, exist_ok=True)
        Path(f"retrieve_results/{self._retrieval_method.name}").mkdir(
            parents=True, exist_ok=True
        )
        writer = ResultsWriter(results)
        # Store JSON in rank_results to a file
        writer.write_in_json_format(
            f"retrieve_results/{self._retrieval_method.name}/retrieve_results_{self._dataset}.json"
        )
        # Store the QRELS of the dataset if specified
        if store_qrels:
            Path("qrels/").mkdir(parents=True, exist_ok=True)
            with open(f"qrels/qrels_{self._dataset}.json", "w") as f:
                json.dump(self._qrels, f, indent=2)
        # Store TRECS if specified
        if store_trec:
            writer.write_in_trec_eval_format(
                f"retrieve_results/{self._retrieval_method.name}/trec_results_{self._dataset}.txt"
            )
        return results


def evaluate_retrievals() -> None:
    from rank_llm.evaluation.trec_eval import EvalFunction

    for dataset in ["dl19", "dl20", "dl21", "dl22"]:
        for retrieval_method in RetrievalMethod:
            if retrieval_method == RetrievalMethod.UNSPECIFIED:
                continue
            file_name = (
                f"retrieve_results/{retrieval_method.name}/trec_results_{dataset}.txt"
            )
            if not os.path.isfile(file_name):
                continue
            EvalFunction.eval(["-c", "-m", "ndcg_cut.10", TOPICS[dataset], file_name])
            EvalFunction.eval(
                ["-c", "-m", "map_cut.100", "-l2", TOPICS[dataset], file_name]
            )


def main():
    for dataset in ["dl19", "dl20", "dl21", "dl22", "news", "covid"]:
        for retrieval_method in RetrievalMethod:
            if retrieval_method in [
                RetrievalMethod.UNSPECIFIED,
                RetrievalMethod.REP_LLAMA,
            ]:
                continue
            retriever = PyseriniRetriever(dataset, retrieval_method)
            retriever.retrieve_and_store()
    evaluate_retrievals()


if __name__ == "__main__":
    main()

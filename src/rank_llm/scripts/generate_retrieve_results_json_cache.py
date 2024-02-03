"""
Genereate retrieve results json cache for a TREC run file.

TREC run file format:
qid Q0 docid rank score run_id

Corpus format:
did\tcontent\n

Query format:
qid\tquery\n

Output format:
[
    {
    "query": "how long is life cycle of flea",
    "hits": [
      {
        "content": "...",
        "qid": 264014,
        "docid": "5611210",
        "rank": 1,
        "score": 15.780599594116211
      },
      {
        "content": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day. The flea egg stage is the beginning of the flea cycle. This part of the flea cycle represents a little more than one third of the flea population at any given time. Depending on the temperature and humidity of the environment the egg can take from two to six days to hatch.",
        "qid": 264014,
        "docid": "6641238",
        "rank": 2,
        "score": 15.090800285339355
      }, ...
      }
      ...
]

Usage:
    python3 scripts/generate_retrieve_results_json_cache.py --trec_file <run_file_path> \
        --collection_file <collection_file_path> \
        --query_file <query_file_path> \
        --output_file <output_file_path>
Example:
    python3 scripts/generate_retrieve_results_json_cache.py --trec_file retrieve_results/BM25/trec_results_train_random_n1000.txt \
        --collection_file /store/scratch/rpradeep/msmarco-v1/collections/official/collection.tsv \
        --query_file /store/scratch/rpradeep/msmarco-v1/collections/official/queries.train.tsv \
        --output_file retrieve_results/BM25/retrieve_results_train_random_n1000.json \
        --topk 20
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from pyserini.search import LuceneSearcher, get_qrels, get_topics
from tqdm import tqdm

from rank_llm.topics_dict import TOPICS

sys.path.append(os.getcwd())


def load_trec_file(file_path, topk=100, qrels=None):
    data = defaultdict(list)
    with open(file_path, "r") as file:
        for line in file:
            qid, _, docid, rank, score, _ = line.strip().split()
            if type(list(qrels.items())[0][0]) is int:
                qid = int(qid)
            if qrels is not None and qid not in qrels:
                continue
            if int(rank) > topk:
                continue
            data[qid].append(
                {"qid": qid, "docid": docid, "rank": int(rank), "score": float(score)}
            )
    print(f"Loaded run for {len(data)} queries from {file_path}")
    return data


def load_tsv_file(file_path):
    examples = {}
    with open(file_path, "r") as file:
        for line in file:
            exid, text = line.strip().split("\t")
            examples[exid] = text
    return examples


def load_pyserini_indexer(collection_file, trec_data, topk):
    examples = {}
    index_reader = LuceneSearcher.from_prebuilt_index(collection_file)
    for qid, hits in tqdm(trec_data.items()):
        rank = 0
        for hit in hits:
            rank += 1
            if rank > topk:
                break
            document = index_reader.doc(hit["docid"])
            content = json.loads(document.raw())
            if "title" in content:
                # content = (
                #     "Title: " + content["title"] + " " +
                #     "Content: " + content["text"]
                # )
                content = content["title"].strip() + ". " + content["text"]
            elif "contents" in content:
                content = content["contents"]
            else:
                content = content["passage"]
            examples[hit["docid"]] = content
    print(f"Loaded {len(examples)} examples from {collection_file}")
    return examples


def generate_retrieve_results(
    trec_file, collection_file, query_file, topk=100, output_trec_file=None
):
    if query_file in TOPICS:
        if TOPICS[query_file] == "dl22-passage":
            query_data = get_topics("dl22")
        elif TOPICS[query_file] == "dl21-passage":
            query_data = get_topics("dl21")
        else:
            query_data = get_topics(TOPICS[query_file])
        for qid, query in query_data.items():
            query_data[qid] = query["title"]
        qrels = get_qrels(TOPICS[query_file])
        print(f"Loaded {len(query_data)} queries from {query_file}")
        print(f"Loaded {len(qrels)} qrels from {query_file}")
    else:
        query_data = load_tsv_file(query_file)
        qrels = None
    trec_data = load_trec_file(trec_file, topk, qrels)

    if collection_file in [
        "msmarco-v2-passage",
        "beir-v1.0.0-trec-covid.flat",
        "beir-v1.0.0-trec-news.flat",
    ]:
        collection_data = load_pyserini_indexer(collection_file, trec_data, topk)
    else:
        collection_data = load_tsv_file(collection_file)

    results = []
    for qid, hits in tqdm(trec_data.items()):
        if qid not in query_data:
            qid = int(qid)
        query = query_data[qid]
        for hit in hits:
            hit["content"] = collection_data[hit["docid"]]
        results.append({"query": query, "hits": hits})
    if output_trec_file is not None:
        with open(output_trec_file, "w") as file:
            for result in results:
                for hit in result["hits"]:
                    file.write(
                        f"{hit['qid']} Q0 {hit['docid']} {hit['rank']} {hit['score']} run_id\n"
                    )
    return results


def write_output_file(output_file_path, data):
    with open(output_file_path, "w") as file:
        json.dump(data, file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_file", required=True)
    parser.add_argument("--collection_file", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--output_trec_file", type=str, default=None)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    results = generate_retrieve_results(
        args.trec_file,
        args.collection_file,
        args.query_file,
        args.topk,
        args.output_trec_file,
    )
    write_output_file(args.output_file, results)


if __name__ == "__main__":
    main()

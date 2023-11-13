'''
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
    python generate_retrieve_results_json_cache.py --trec_file <run_file_path> \
        --collection_file <collection_file_path> \
        --query_file <query_file_path> \
        --output_file <output_file_path>
Example:

'''

import argparse
import json
import os
import sys
from collections import defaultdict

from tqdm import tqdm

sys.path.append(os.getcwd())

def load_trec_file(file_path):
    data = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            qid, _, docid, rank, score, _ = line.strip().split()
            data[qid].append({
                'qid': qid,
                'docid': docid,
                'rank': int(rank),
                'score': float(score)
            })
    return data
def load_tsv_file(file_path):
    examples = {}
    with open(file_path, 'r') as file:
        for line in file:
            exid, text = line.strip().split('\t')
            examples[exid] = text
    return examples

def generate_retrieve_results(trec_file, collection_file, query_file):
    trec_data = load_trec_file(trec_file)
    collection_data = load_tsv_file(collection_file)
    query_data = load_tsv_file(query_file)

    results = []
    for qid, hits in tqdm(trec_data.items()):
        query = query_data[qid]
        for hit in hits:
            hit['content'] = collection_data[hit['docid']]
        results.append({
            'query': query,
            'hits': hits
        })     
    return results
        

def write_output_file(output_file_path, data):
    with open(output_file_path, 'w') as file:
        json.dump(data, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trec_file', required=True)
    parser.add_argument('--collection_file', required=True)
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()

    results = generate_retrieve_results(args.trec_file, args.collection_file, args.query_file)
    write_output_file(args.output_file, results)

if __name__ == "__main__":
    main()

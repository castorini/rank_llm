import hashlib
import json
import os


def generate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_size(file_path):
    return os.path.getsize(file_path)


def generate_hits_info(base_path):
    hits_info = {}
    METHOD_MAPPING = {
        "BM25": "BM25",
        "BD_BERT_KD_TASB": "DistilBERT KD TASB",
        "OPEN_AI_ADA2": "OpenAI ADA2",
        "SPLADE_P_P_ENSEMBLE_DISTIL": "SPLADE++ CoCondenser-EnsembleDistil",
        "BM25_RM3": "BM25 + RM3",
        "REP_LLAMA": "repLLaMA",
    }
    DATASET_MAPPING = {
        "msmarco_passage": "MS MARCO Passage",
        "msmarco_document": "MS MARCO Document",
        "robust04": "Robust04",
        "core17": "Core17",
        "core18": "Core18",
        "dl19": "TREC DL19",
        "dl20": "TREC DL20",
        "dl21": "TREC DL21",
        "dl22": "TREC DL22",
        "dl23": "TREC DL23",
        "covid": "TREC COVID",
        "news": "TREC News",
        "noveleval": "NovelEval",
        "msmarco-v2.1-doc-segmented.bm25.rag24.raggy-dev": "TREC24-Raggy Dev",
        "msmarco-v2.1-doc-segmented.bm25.rag24.researchy-dev": "TREC24-Researchy Dev",
    }
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, base_path)
                print(f"Processing {relative_path}...")
                method = relative_path.split("/")[1]
                if method in METHOD_MAPPING:
                    method = METHOD_MAPPING[method]
                dataset = relative_path.split("/")[2]
                k = (
                    dataset[dataset.rfind("_top") + len("_top") :]
                    .replace(".jsonl", "")
                    .strip()
                )
                dataset = dataset[
                    dataset.find("retrieve_results_")
                    + len("retrieve_results_") : dataset.rfind("_top")
                ]
                if dataset in DATASET_MAPPING:
                    dataset = DATASET_MAPPING[dataset]
                print(f"Method: {method}, Dataset: {dataset}, k: {k}")
                hits_info[relative_path] = {
                    "description": f"Top-{k} {method} Results for the {dataset} task. (Cached JSONL format)",
                    "urls": [
                        f"https://github.com/castorini/rank_llm_data/raw/main/{relative_path}"
                    ],
                    "md5": generate_md5(file_path),
                    "size (bytes)": get_file_size(file_path),
                    "downloaded": False,
                }
            elif file.endswith(".txt") or file.endswith(".trec"):
                ending = ".txt" if file.endswith(".txt") else ".trec"
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, base_path)
                print(f"Processing {relative_path}...")
                method = relative_path.split("/")[1]
                if method in METHOD_MAPPING:
                    method = METHOD_MAPPING[method]
                dataset = relative_path.split("/")[2]
                k = (
                    dataset[dataset.rfind("_top") + len("_top") :]
                    .replace(ending, "")
                    .strip()
                )
                dataset = dataset[
                    dataset.find("trec_results_")
                    + len("trec_results_") : dataset.rfind("_top")
                ]
                if dataset in DATASET_MAPPING:
                    dataset = DATASET_MAPPING[dataset]
                print(f"Method: {method}, Dataset: {dataset}, k: {k}")
                hits_info[relative_path] = {
                    "description": f"Top-{k} {method} Results for the {dataset} task. (TREC format)",
                    "urls": [
                        f"https://github.com/castorini/rank_llm_data/raw/main/{relative_path}"
                    ],
                    "md5": generate_md5(file_path),
                    "size (bytes)": get_file_size(file_path),
                    "downloaded": False,
                }

    # Sort the dictionary by primary keys
    sorted_hits_info = dict(sorted(hits_info.items()))

    return sorted_hits_info


base_path = "./"  # Adjust the base path as necessary
hits_info = generate_hits_info(base_path)

with open("repo_info.py", "w") as f:
    # Write it in form HITS_INFO = {...} (Make sure nice indent)
    # Do not use JSON dump as it will not be in the correct format
    f.write("HITS_INFO = {\n")
    for key, value in hits_info.items():
        key = key.replace("retrieve_results/", "")
        f.write(f"    {json.dumps(key)}: ")
        f.write("{\n")
        for k, v in value.items():
            if k == "urls":
                f.write(f"        {json.dumps(k)}: [\n")
                for url in v:
                    f.write(f"            {json.dumps(url)},\n")
                f.write("        ],\n")
            elif k != "downloaded":
                f.write(f"        {json.dumps(k)}: {json.dumps(v, indent=4)},\n")
            else:
                f.write(f"        {json.dumps(k)}: {v},\n")
        f.write("    },\n")
    f.write("}\n")

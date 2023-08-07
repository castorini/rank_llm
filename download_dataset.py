import pandas as pd
from rank_gpt import run_retriever, sliding_windows, write_eval_file
from pyserini.search import LuceneSearcher, get_topics, get_qrels
from tqdm import tqdm
import tempfile
import os
import json
import shutil
from pathlib import Path
from topics_dict import TOPICS
from indices_dict import INDICES

openai_key = os.environ.get("OPENAI_API_KEY", None)

# for data in ['dl19', 'dl20', 'covid', 'nfc', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']:
for data in ["signal", "news"]:
    print("#" * 20)
    print(f"Evaluation on {data}")
    print("#" * 20)

    # Retrieve passages using pyserini BM25.
    # Get a specific doc:
    # * searcher.num_docs
    # * json.loads(searcher.object.reader.document(4).fields[1].fieldsData) -> {"id": "1", "contents": ""}
    searcher = LuceneSearcher.from_prebuilt_index(INDICES[data])
    topics = get_topics(TOPICS[data] if data != "dl20" else "dl20")
    qrels = get_qrels(TOPICS[data])

    # Create a folder for the dataset
    data_folder = Path(__file__).parent / "data" / data
    data_folder.mkdir(exist_ok=True, parents=True)

    # Store JSON in rank_results to a file
    with open(data_folder / "queries.jsonl", "w") as f:
        for key, value in topics.items():
            f.write(
                json.dumps({"_id": str(key), "text": value["title"]}, ensure_ascii=False) + "\n"
            )
    # Store the QRELS of the dataset
    (data_folder / "qrels").mkdir(exist_ok=True, parents=True)
    with open(data_folder / "qrels" / "test.tsv", "w") as f:
        qrels_list = []
        for query, value in qrels.items():
            for doc, rel in value.items():
                qrels_list += [[query, doc, rel]]
        qrels_data = pd.DataFrame(
            qrels_list, columns=["query-id", "corpus-id", "score"]
        )
        qrels_data.to_csv(f, sep="\t", index=False, header=True)
    # Retrieve all the documents in the corpus
    with open(data_folder / "corpus.jsonl", "w") as f:
        for i in tqdm(range(searcher.num_docs)):
            doc = json.loads(searcher.object.reader.document(i).fields[1].fieldsData)
            doc["_id"] = str(doc["_id"])
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")



# for data in ['mrtydi-ar', 'mrtydi-bn', 'mrtydi-fi', 'mrtydi-id', 'mrtydi-ja', 'mrtydi-ko', 'mrtydi-ru', 'mrtydi-sw', 'mrtydi-te', 'mrtydi-th']:
#     print('#' * 20)
#     print(f'Evaluation on {data}')
#     print('#' * 20)

#     # Retrieve passages using pyserini BM25.
#     try:
#         searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
#         topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
#         qrels = get_qrels(THE_TOPICS[data])
#         rank_results = run_retriever(topics, searcher, qrels, k=100)
#         rank_results = rank_results[:100]

#         # Store JSON in rank_results to a file
#         with open(f'rank_results_{data}.json', 'w') as f:
#             json.dump(rank_results, f, indent=2)
#         # Store the QRELS of the dataset
#         with open(f'qrels_{data}.json', 'w') as f:
#             json.dump(qrels, f, indent=2)
#     except:
#         print(f'Failed to retrieve passages for {data}')

#     # # Run sliding window permutation generation
#     # new_results = []
#     # for item in tqdm(rank_results):
#     #     new_item = sliding_windows(item, rank_start=0, rank_end=100, window_size=20, step=10,
#     #                                model_name='gpt-3.5-turbo', openai_key=openai_key)
#     #     new_results.append(new_item)

#     # # Evaluate nDCG@10
#     # from trec_eval import EvalFunction

#     # temp_file = tempfile.NamedTemporaryFile(delete=False).name
#     # write_eval_file(new_results, temp_file)
#     # EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data], temp_file])
#     #     # Rename the output file to a better name
#     # shutil.move(output_file, f'eval_{data}.txt')

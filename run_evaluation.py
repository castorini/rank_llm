from rank_llm import RankLLM
from pyserini_retriever import PyseriniRetriever
from tqdm import tqdm
import tempfile
import os
import json
import shutil
from topics_dict import TOPICS
from indices_dict import INDICES

openai_key = os.environ.get("OPENAI_API_KEY", None)

for data in TOPICS.keys():
# for data in ['signal', 'news', 'robust04']:

    print('#' * 20)
    print(f'Evaluation on {data}')
    print('#' * 20)

    # Retrieve passages using pyserini BM25.
    # Get a specific doc: 
    # * searcher.num_docs
    # * json.loads(searcher.object.reader.document(4).fields[1].fieldsData) -> {"id": "1", "contents": ""}
    try:
        retriever = PyseriniRetriever(data)
        retriever.retrieve_and_store(100)
    except Exception as e:
        print(f'Failed to retrieve passages for {data}: {e}')
    # # Run sliding window permutation generation
    # new_results = []
    # for item in tqdm(rank_results):
    #     new_item = sliding_windows(item, rank_start=0, rank_end=10, window_size=20, step=10,
    #                                model_name='gpt-3.5-turbo', openai_key=openai_key)
    #     new_results.append(new_item)

    # # Evaluate nDCG@10
    # from trec_eval import EvalFunction

    # # Create an empty text file to write results, and pass the name to eval
    # output_file = tempfile.NamedTemporaryFile(delete=False).name
    # write_eval_file(new_results, output_file)
    # EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data], output_file])
    # # Rename the output file to a better name
    # shutil.move(output_file, f'eval_{data}.txt')



for data in ['mrtydi-ar', 'mrtydi-bn', 'mrtydi-fi', 'mrtydi-id', 'mrtydi-ja', 'mrtydi-ko', 'mrtydi-ru', 'mrtydi-sw', 'mrtydi-te', 'mrtydi-th']:
    print('#' * 20)
    print(f'Evaluation on {data}')
    print('#' * 20)

    # Retrieve passages using pyserini BM25.
    try:
        retriever = PyseriniRetriever(data)
        retriever.retrieve_and_store(100)

    except:
        print(f'Failed to retrieve passages for {data}')

    # # Run sliding window permutation generation
    # new_results = []
    # for item in tqdm(rank_results):
    #     new_item = sliding_windows(item, rank_start=0, rank_end=100, window_size=20, step=10,
    #                                model_name='gpt-3.5-turbo', openai_key=openai_key)
    #     new_results.append(new_item)

    # # Evaluate nDCG@10
    # from trec_eval import EvalFunction

    # temp_file = tempfile.NamedTemporaryFile(delete=False).name
    # write_eval_file(new_results, temp_file)
    # EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data], temp_file])
    #     # Rename the output file to a better name
    # shutil.move(output_file, f'eval_{data}.txt')

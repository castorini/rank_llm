import json
from tqdm import tqdm
from pyserini.search import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from pathlib import Path
from indices_dict import INDICES
from topics_dict import TOPICS

class PyseriniRetriever:
    def __init__(self, dataset:str):
        if dataset not in INDICES:
            raise ValueError("dataset %s not in INDICES" % dataset)
        self.searcher = LuceneSearcher.from_prebuilt_index(INDICES[dataset])
        if not self.searcher:
            raise ValueError("invalid dataset name: %s" %dataset)
        if dataset not in TOPICS:
            raise ValueError("dataset %s not in TOPICS" % dataset)
        if dataset == 'dl20':
            topics_key = dataset
        else:
            topics_key = TOPICS[dataset]
        self.topics = get_topics(topics_key)
        self.qrels = get_qrels(TOPICS[dataset])
        self.dataset = dataset

    def _retrieve_query(self, query:str, ranks, k:int, qid=None):
        hits = self.searcher.search(query, k=k)
        ranks.append({'query': query, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(self.searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})

    def retrieve(self, k=100, qid=None):
        ranks = []
        if isinstance(self.topics, str):
            self._retrieve_query(self.topics, ranks, k, qid)
            return ranks[-1]

        for qid in tqdm(self.topics):
            if qid in self.qrels:
                query = self.topics[qid]['title']
                self._retrieve_query(query, ranks, k, qid)
        return ranks
    
    def num_queries(self):
        if isinstance(self.topics, str):
            return 1
        return len(self.topics)

    def retrieve_and_store(self, k=100, qid=None, store_qrels:bool=True):
        results = self.retrieve(k, qid)
        Path("retrieve_results/").mkdir(parents=True, exist_ok=True)
        # Store JSON in rank_results to a file
        with open(f'retrieve_results/retrieve_results_{self.dataset}.json', 'w') as f:
            json.dump(results, f, indent=2)
        # Store the QRELS of the dataset if specified
        if store_qrels:
            Path("qrels/").mkdir(parents=True, exist_ok=True)
            with open(f'qrels/qrels_{self.dataset}.json', 'w') as f:
                json.dump(self.qrels, f, indent=2)


def main():
    dataset = 'dl19'
    retriever = PyseriniRetriever(dataset)
    retriever.retrieve_and_store()


if __name__ == '__main__':
    main()

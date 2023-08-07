import json
from torch.utils.data import Dataset
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AdamW
import torch
from tqdm import tqdm
from rank_loss import RankLoss
import numpy as np
import os
import argparse
import tempfile
import copy
import sys


class RerankData(Dataset):
    def __init__(self, data, tokenizer, neg_num=20, label=True):
        self.data = data
        self.tokenizer = tokenizer
        self.neg_num = neg_num
        self.label = label
        if not label:
            self.neg_num += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        query = item['query']

        if self.label:
            pos = [str(item['positive_passages'][0]['text'])]
            pos_id = [psg['docid'] for psg in item['positive_passages']]
            neg = [str(psg['text']) for psg in item['retrieved_passages'] if psg['docid'] not in pos_id][:self.neg_num]
        else:
            pos = []
            neg = [str(psg['text']) for psg in item['retrieved_passages']][:self.neg_num]
        neg = neg + ['<padding_passage>'] * (self.neg_num - len(neg))
        passages = pos + neg
        return [query] * len(passages), passages

    def collate_fn(self, data):
        query, passages = zip(*data)
        query = sum(query, [])
        passages = sum(passages, [])
        features = self.tokenizer(query, passages, padding=True, truncation=True, return_tensors="pt",
                                  max_length=500)
        return features


def receive_response(data, responses):
    def clean_response(response: str):
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def remove_duplicate(response):
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    new_data = []
    for item, response in zip(data, responses):
        response = clean_response(response)
        response = [int(x) - 1 for x in response.split()]
        response = remove_duplicate(response)
        passages = item['retrieved_passages']
        original_rank = [tt for tt in range(len(passages))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        new_passages = [passages[ii] for ii in response]
        new_data.append({'query': item['query'],
                         'positive_passages': item['positive_passages'],
                         'retrieved_passages': new_passages})
    return new_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='microsoft/deberta-v3-base')
    parser.add_argument('--loss', type=str, default='rank_net')
    parser.add_argument('--data', type=str, default='data/marco-train-10k.jsonl')
    parser.add_argument('--save_path', type=str, default='out/deberta-rank_net')
    parser.add_argument('--permutation', type=str, default='marco-train-10k-gpt3.5.json')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


def train(args):
    model_name = args.model
    loss_type = args.loss
    data_path = args.data
    save_path = args.save_path
    permutation = args.permutation

    accelerator = Accelerator(gradient_accumulation_steps=8)
    neg_num = 19

    data = [json.loads(line) for line in open(data_path)]
    print('Num of training samples', len(data))

    # Create cross encoder model
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Load data and permutation
    data = [json.loads(line) for line in open(data_path)]
    response = json.load(open(permutation))
    data = receive_response(data, response)
    dataset = RerankData(data, tokenizer, neg_num=neg_num, label=False)

    # Prepare data loader
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn,
                                              batch_size=1, shuffle=True, num_workers=0)
    optimizer = AdamW(model.parameters(), 5e-5)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    # Prepare loss function
    loss_function = getattr(RankLoss, loss_type)

    # Train for 3 epoch
    for epoch in range(3):
        accelerator.print(f'Training {save_path} {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            with accelerator.accumulate(model):
                out = model(**batch)
                logits = out.logits
                logits = logits.view(-1, neg_num + 1)

                y_true = torch.tensor([[1 / (i + 1) for i in range(logits.size(1))]] * logits.size(0)).cuda()
                loss = loss_function(logits, y_true)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
                loss_report.append(accelerator.gather(loss).mean().item())
            tk0.set_postfix(loss=sum(loss_report) / len(loss_report))
        accelerator.wait_for_everyone()

    # Save model
    unwrap_model = accelerator.unwrap_model(model)
    os.makedirs(save_path, exist_ok=True)
    unwrap_model.save_pretrained(save_path)

    return model, tokenizer


def eval_on_benchmark(args, model=None, tokenizer=None):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
    from pyserini_retriever import PyseriniRetriever
    from rank_gpt import receive_permutation, write_eval_file
    from trec_eval import EvalFunction
    from topics_dict import TOPICS

    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
        model = model.cuda()

    model.eval()

    for data in ['dl19', 'dl20', 'covid', 'nfc', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']:
        print()
        print('#' * 20)
        print(f'Now eval [{data}]')
        print('#' * 20)

        retriever = PyseriniRetriever(data)
        rank_results = retriever.retrieve(k=100)

        reranked_data = []
        for item in tqdm(rank_results):
            q = item['query']
            passages = [psg['content'] for i, psg in enumerate(item['hits'])][:100]
            if len(passages) == 0:
                reranked_data.append(item)
                continue
            features = tokenizer([q] * len(passages), passages, padding=True, truncation=True, return_tensors="pt",
                                 max_length=500)
            features = {k: v.cuda() for k, v in features.items()}
            with torch.no_grad():
                scores = model(**features).logits
                normalized_scores = [float(score[0]) for score in scores]
            ranked = np.argsort(normalized_scores)[::-1]
            response = ' > '.join([str(ss + 1) for ss in ranked])
            reranked_data.append(receive_permutation(item, response, rank_start=0, rank_end=100))

        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        write_eval_file(reranked_data, temp_file)
        EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', TOPICS[data], temp_file])


if __name__ == '__main__':
    args = parse_args()
    model, tokenizer = None, None
    if args.do_train:
        model, tokenizer = train(args)
    if args.do_eval:
        eval_on_benchmark(args, model, tokenizer)

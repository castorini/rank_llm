import torch
import torch.utils.data
import random
import json
import numpy as np
from .options import Options

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_passages=None,
                 start_pos=0,
                 question_prefix='question:',
                 passage_prefix='context:',
                 passage_numbering=False):
        self.data = data
        self.n_passages = n_passages
        self.start_pos = start_pos
        self.question_prefix = question_prefix
        self.passage_prefix = passage_prefix
        self.passage_numbering = passage_numbering

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']

        if 'ctxs' in example and self.n_passages is not None:
            # add dummy contexts when there are not enough
            while len(example['ctxs']) < self.start_pos+self.n_passages:
                example['ctxs'].append({'text': ""})
            
            contexts = np.array(example['ctxs'][self.start_pos:self.start_pos+self.n_passages])

            if self.passage_numbering:
                f = self.passage_prefix + " [{}] {}"
                passages = []
                passage_id = 1
                for c in contexts:
                    passages.append(f.format(passage_id, c['text']))
                    passage_id+=1
            else:
                f = self.passage_prefix + " {}"
                passages = np.array([f.format(c['text']) for c in contexts])
            
        else:
            passages = None
        return {
            'index' : index,
            'question' : question,
            'passages' : passages,
        }

def encode_passages(batch_text_passages, tokenizer, max_length, batch_size, n_passages):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length', 
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=32, batch_size=1, n_passages=100, suffix=''):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.batch_size = batch_size
        self.n_passages = n_passages
        self.suffix = suffix

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t + self.suffix for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        query = [example['question'] for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength,
                                                     self.batch_size,
                                                     self.n_passages)

        return (index, passage_ids, passage_masks, query)

def load_data(data_path):
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        examples.append(example)

    if data_path.endswith('.jsonl'):
        data.close()

    return examples
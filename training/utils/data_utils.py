import os
import copy
import logging
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from ftfy import fix_text

max_psg_num = 20
START_IDX = ord('A')
IGNORE_INDEX = -100

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return input_ids, labels, sources_tokenized["input_ids_lens"]

class RankingDataset(Dataset):
    """Dataset for ranking tasks."""
    def __init__(self, raw_data, model_tokenizer, type) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.tokenizer.padding_side="left"
        self.type = type
        self.system_message_supported = "system" in self.tokenizer.chat_template
    
    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]
        
        # Validate conversation structure
        if len(conversation) < 3:
            raise ValueError(f"Invalid conversation format at index {index}")
            
        sys_msg = conversation[0]['value']
        input_context = conversation[1]['value']
        target_generation = conversation[2]["value"]

        if self.system_message_supported:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": input_context}
            ]
        else:
            messages = [
                {"role": "user", "content": sys_msg + "\n " + input_context}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += "["
        prompt = fix_text(prompt)

        if self.type == "train":
            label_map = {}
            label_rank = 0
            for token in target_generation:
                if token.isalpha() and ord(token) >= START_IDX:
                    label_map[token] = label_rank
                    label_rank += 1
            
            label = [label_map[chr(c)] for c in range(START_IDX, START_IDX+len(label_map))]

        elif self.type == "eval":
            label = [self.raw_data[index]["id"]] + self.raw_data[index]["docids"] + self.raw_data[index]["scores"]
        else:
            raise Exception("Invalid run type specified for Dataset. Choose from ['train', 'eval']")
        return prompt, label
    
    def __len__(self):
        return len(self.raw_data)

class GenerationDataset(Dataset):
    """Dataset for generation tasks."""
    def __init__(self, raw_data, model_tokenizer, combined=False) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.combined = combined
        self.system_message_supported = "system" in self.tokenizer.chat_template
    
    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]
        sys_msg = conversation[0]['value']
        input_context = conversation[1]['value']
        label = conversation[2]["value"]
        label += self.tokenizer.eos_token
        
        if self.system_message_supported:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": input_context}
            ]
        else:
            messages = [
                {"role": "user", "content": sys_msg + "\n " + input_context}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = fix_text(prompt)
        if self.combined:
            label_map = {}
            label_rank = 0
            for token in conversation[2]["value"]:
                if token.isalpha():
                    label_map[token] = label_rank
                    label_rank += 1
            
            rank_label = [label_map[chr(c)] for c in range(START_IDX, START_IDX+len(label_map))]
            return prompt, label, rank_label
        else:
            return prompt, label
    
    def __len__(self):
        return len(self.raw_data)

def ranking_collate_fn(data, tokenizer):
    """Collate function for ranking datasets."""
    prompts, labels = list(zip(*data))
    tokenized_inputs = tokenizer(prompts, padding="longest", truncation=False, return_tensors="pt")
    return tokenized_inputs, labels

def generation_collate_fn(data, tokenizer):
    """Collate function for generation datasets."""
    prompts, labels = list(zip(*data))
    tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return tokenized_inputs, labels

def combined_collate_fn(data, tokenizer):
    """Collate function for combined ranking and generation datasets."""
    prompts, labels, rank_labels = list(zip(*data))
    tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return tokenized_inputs, labels, rank_labels, source_lens

def load_data(file_path):
    """Load data from a file."""
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def initialize_dataset_and_loader(args, tokenizer):
    """
    Initialize dataset and dataloader based on provided arguments.
    
    Args:
        args: Namespace object containing dataset initialization arguments
        tokenizer: Tokenizer to be used with the dataset
        
    Returns:
        tuple: (dataset, dataloader)
    """
    # Load raw data
    if not os.path.isfile(args.train_dataset_path):
        # Using Hugging Face dataset
        ds = load_dataset(args.train_dataset_path)
        raw_train_data = ds['train']
    else:
        raw_train_data = load_data(args.train_dataset_path)

    # Initialize appropriate dataset and collate function based on objective
    if args.objective == "generation":
        train_dataset = GenerationDataset(raw_train_data, tokenizer)
        train_collate_fn = generation_collate_fn
    elif args.objective == "combined":
        train_dataset = GenerationDataset(raw_train_data, tokenizer, combined=True)
        train_collate_fn = combined_collate_fn
    else:
        train_dataset = RankingDataset(raw_train_data, tokenizer, type="train")        
        train_collate_fn = ranking_collate_fn

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=partial(train_collate_fn, tokenizer=tokenizer),
        batch_size=args.per_device_train_batch_size
    )

    return train_dataset, train_dataloader 
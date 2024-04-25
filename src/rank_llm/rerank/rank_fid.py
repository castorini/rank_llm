import torch
import transformers

torch.manual_seed(0)
transformers.set_seed(0)

import numpy as np
from torch.utils.data import DataLoader, SequentialSampler

import lit5.data
import lit5.model

import copy
import json 
from typing import List, Union, Dict, Any, Tuple

from rank_llm.rerank.rankllm import PromptMode, RankLLM
from rank_llm.result import Result

if opt.write_crossattention_scores:
    from lit5.modeling_t5 import T5ForConditionalGeneration, T5Stack
else:
    from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack


class RankFiDDistill(RankLLM):
    def __init__(
        self,
        model_path: str,
        context_size: int = 300,
        prompt_mode: PromptMode = PromptMode.LiT5,  # Placeholder for actual mode
        window_size: int = 20,
        device: str = 'cuda',
    ) -> None:
        """
         Creates instance of the RankFiDDistill class, a specialized version of RankLLM designed from Lit5-Distill.
        """
        super().__init__(model=model_path, context_size=context_size, prompt_mode=prompt_mode)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.window_size = window_size
        self.device = device
        self.batch_size = 1
        self.stride = 10
        self.answer_maxlength = 100
        self.n_passes = 1
    
    # leave for now, the same code from LiT5-Distill.py
    def evaluate(self, model, dataset, dataloader, tokenizer):
        generated_permutations = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                (idx, passage_ids, passage_mask, query) = batch
                passage_ids = passage_ids.contiguous().view(passage_ids.size(0), -1)
                passage_mask = passage_mask.contiguous().view(passage_mask.size(0), -1)

                outputs = model.generate(
                    input_ids=passage_ids.cuda(),
                    attention_mask=passage_mask.cuda(),
                    max_length=opt.answer_maxlength,
                    do_sample=False
                )

                for k, o in enumerate(outputs):
                    output = tokenizer.decode(o, skip_special_tokens=True)
                    generated_permutations.append(output)
        return generated_permutations

    # leave for now, the same code from LiT5-Distill.py
    def clean_response(self, response: str) -> str:
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response
    
    # leave for now, the same code from LiT5-Distill.py
    def remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def run_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> Tuple[str, int]:
        """
        Run the target language model with a passed in prompt.
        """
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=self.context_size).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.answer_maxlength,
            do_sample=False
        )
        decoded_outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_outputs, outputs.shape[1]

    def create_prompt(self, result: Result, rank_start: int, rank_end: int) -> Tuple[str, int]:
        """
        Create a prompt based on the result and given ranking range.
        """
        passages = [result.hits[i]['content'] for i in range(rank_start, rank_end)]
        prompt = " ".join(passages)
        prompt_length = self.get_num_tokens(prompt)
        return prompt, prompt_length

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """
        Abstract method to calculate the number of tokens contained in the given prompt.
        """
        if isinstance(prompt, str):
            return len(self.tokenizer.encode(prompt))
        elif isinstance(prompt, list):
            return sum(len(self.tokenizer.encode(item['text'])) for item in prompt)
        else:
            raise ValueError("Prompt must be a string or a list of dictionaries with a 'text' key.")

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

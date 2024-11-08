import logging
import math
from typing import List, Tuple
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, MT5Tokenizer, MT5ForConditionalGeneration
from transformers.generation import GenerationConfig

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM

logger = logging.getLogger(__name__)

PREDICTION_TOKENS = {
        'castorini/monot5-base-msmarco':             ['▁false', '▁true'],
        'castorini/monot5-base-msmarco-10k':         ['▁false', '▁true'],
        'castorini/monot5-large-msmarco':            ['▁false', '▁true'],
        'castorini/monot5-large-msmarco-10k':        ['▁false', '▁true'],
        'castorini/monot5-base-med-msmarco':         ['▁false', '▁true'],
        'castorini/monot5-3b-med-msmarco':           ['▁false', '▁true'],
        'castorini/monot5-3b-msmarco-10k':           ['▁false', '▁true'],
        'unicamp-dl/mt5-base-en-msmarco':            ['▁no'   , '▁yes'],
        'unicamp-dl/ptt5-base-pt-msmarco-10k-v2':    ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2':   ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2':['▁não'  , '▁sim'],
        'unicamp-dl/mt5-base-en-pt-msmarco-v2':      ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-mmarco-v2':             ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-en-pt-msmarco-v1':      ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-mmarco-v1':             ['▁no'   , '▁yes'],
        'unicamp-dl/ptt5-base-pt-msmarco-10k-v1':    ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v1':   ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['▁não'  , '▁sim'],
        'unicamp-dl/mt5-3b-mmarco-en-pt':            ['▁'  , '▁true'],
        'unicamp-dl/mt5-13b-mmarco-100k':            ['▁', '▁true'],
        }


class MonoT5(PointwiseRankLLM):
    def __init__(
        self,
        model: str,
        prompt_mode: str = "monot5",
        context_size: int = 512,
        device: str = "cuda",
        batch_size: int = 32,
        dtype: str = "float16"
    ):
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            device=device,
            batch_size=batch_size,
        )

        if model.find("mt5") != -1:
            self._tokenizer = MT5Tokenizer.from_pretrained(model)
            dtype = torch.float16 if dtype == "float16" else torch.float32
            self._llm = MT5ForConditionalGeneration.from_pretrained(model, torch_dtype=dtype).to(self._device)
        elif model.find("monot5") != -1:
            self._tokenizer = T5Tokenizer.from_pretrained(model)
            self._llm = T5ForConditionalGeneration.from_pretrained(model).to(self._device)
        self._context_size = context_size

    @staticmethod
    def get_prediction_tokens(pretrained_model_name_or_path: str, tokenizer):
        if pretrained_model_name_or_path in PREDICTION_TOKENS:
            token_false, token_true = PREDICTION_TOKENS[pretrained_model_name_or_path]
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id  = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id
        else:
            raise Exception(f"We don't know the indexes for the non-relevant/relevant tokens for\
                    the checkpoint {pretrained_model_name_or_path} and you did not provide any.")
        pass
    def run_llm_batched(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], List[int], List[float]]:
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens()
        gen_cfg.min_new_tokens = self.num_output_tokens()
        gen_cfg.output_scores = True
        gen_cfg.return_dict_in_generate = True
        gen_cfg.do_sample = False

        all_outputs = []
        all_output_token_counts = []
        all_scores = []

        batch_prompts = prompts

        token_prompts = self._tokenizer(
            batch_prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(self._device)

        token_prompts = token_prompts["input_ids"]

        batch_outputs = self._llm.generate(token_prompts, generation_config=gen_cfg)

        batch_output_ids = batch_outputs.sequences
        batch_logits = batch_outputs.scores

        batch_outputs = [
            self._tokenizer.decode(
                single_token_sequence,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            for single_token_sequence in batch_output_ids
        ]

        for logit_tensor in batch_logits[0]:
            token_false_id, token_true_id = self.get_prediction_tokens(self._model, self._tokenizer)
            pos_index = logit_tensor[token_true_id]
            neg_index = logit_tensor[token_false_id]
            #wanted_logit = logit_tensor.find("")
            #print(f"current False_logit is {neg_index}")
            #print(f"wanted false logit can be found at {wanted_logit}")

            score = math.exp(pos_index) / (
                math.exp(pos_index) + math.exp(neg_index)
            )
            all_scores.append(score)
            all_output_token_counts.append(self.num_output_tokens)

        all_outputs.extend(batch_outputs)

        return all_outputs, all_output_token_counts, all_scores



    def run_llm(self, prompt: str) -> Tuple[str, int, float]:
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens()
        gen_cfg.min_new_tokens = self.num_output_tokens()
        gen_cfg.output_scores = True
        gen_cfg.return_dict_in_generate = True
        gen_cfg.do_sample = False

        token_prompt = self._tokenizer.encode(prompt, return_tensors="pt").to(
            self._device
        )
        output = self._llm.generate(token_prompt, generation_config=gen_cfg)
        output_ids = output.sequences
        logits = output.scores

        if self._llm.config.is_encoder_decoder:
            output_ids = output_ids[0]
            output_ids = output_ids[1:]

        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        truth_logit = logits[0][0][1176]
        false_logit = logits[0][0][6136]
        score = math.exp(truth_logit) / (math.exp(truth_logit) + math.exp(false_logit))

        return outputs, output_ids.size(0), score

    def num_output_tokens(self) -> int:
        return 1

    def create_prompt(self, result: Result, index: int) -> Tuple[str, int]:
        query = result.query.text
        input = f"Query: {query} Document: {self.convert_doc_to_prompt_content(result.candidates[index].doc, max_length=self._context_size)}"
        prompt = (
            self._tokenizer.decode(
                self._tokenizer.encode(input)[: (self._context_size - 32)]
            )[:-4]
            + " Relevant: "
        )
        prompt = prompt.replace("<unk>", "")

        return prompt, self.get_num_tokens(prompt)

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

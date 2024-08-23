import logging
import math
from typing import List, Tuple
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM

logger = logging.getLogger(__name__)


class BGE_RERANKER_V2(PointwiseRankLLM):
    def __init__(
        self,
        model: str,
        prompt_mode: str = "bge-reranker-v2",
        context_size: int = 512,
        device: str = "cuda",
        batch_size: int = 32,
        use_bf16: bool = False
    ):
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            device=device,
            batch_size=batch_size,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model,
            trust_remote_code=True,
            padding_side="left"
        )
        self._llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32
        ).to(self._device)
        self._llm.eval()
        self._context_size = context_size

    def get_inputs(pairs, tokenizer, prompt:str=None, max_length:int=1024):
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        print(tokenizer)
        prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)['input_ids']
        print(prompt_inputs)
        sep_inputs = tokenizer(sep, return_tensors="pt", add_special_tokens=False)['input_ids']
        inputs = []
        for query, passage in pairs:
            query_inputs = tokenizer(f'A: {query}',
                return_tensors="pt",
                add_special_tokens=False,
                max_length=max_length * 3 // 4,
                truncation=True)
            passage_inputs = tokenizer(f'B: {passage}',
                return_tensors="pt",
                add_special_tokens=False,
                max_length=max_length,
                truncation=True)
            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
        return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

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

        pairs = [[prompt.split("Document: ").pop(0).replace("<s> Query: ", ""), prompt.split("Document: ").pop().replace("Relevant:", "")] for prompt in prompts]

        with torch.no_grad():
            if "base" in self._model or "large" in self._model or "m3" in self._model:

                token_prompts = self._tokenizer(
                    pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
                ).to(self._device)

                token_prompts = token_prompts["input_ids"]

                batch_outputs = self._llm.generate(token_prompts, generation_config=gen_cfg)

            elif "gemma" in self._model:
                idk = 0

            elif "minicpm-layerwise" in self._model:
                inputs = self.get_inputs(pairs, self._tokenizer)
                inputs = inputs.to(self._device)
                #all_scores = self._llm(**inputs, return_dict=True, cutoff_layers=[28])
                batch_outputs = self._llm.generate(**inputs, generation_config=gen_cfg)
                


            else:
                raise ValueError("Given bge model doesn't exist or isn't supported in rank_llm.")

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
            truth_logit = logit_tensor[1176]
            false_logit = logit_tensor[6136]
            score = math.exp(truth_logit) / (
                math.exp(truth_logit) + math.exp(false_logit)
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

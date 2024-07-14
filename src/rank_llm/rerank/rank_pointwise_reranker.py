import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
from fastchat.model import get_conversation_template, load_model
from ftfy import fix_text
from transformers.generation import GenerationConfig

try:
    from vllm import LLM, SamplingParams
except:
    LLM = None
    SamplingParams = None

from rank_llm.data import Result
from rank_llm.rerank.rankllm import PromptMode, RankLLM
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer


class RankPointwise(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int = 512,
        prompt_mode: PromptMode = PromptMode.MONO,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
        batched: bool = False,
    ) -> None:
        """
         Creates instance of the RankListwiseOSLLM class, an extension of RankLLM designed for performing listwise ranking of passages using
         a specified language model. Advanced configurations are supported such as GPU acceleration, variable passage
         handling, and custom system messages for generating prompts.

         Parameters:
         - model (str): Identifier for the language model to be used for ranking tasks.
         - context_size (int, optional): Maximum number of tokens that can be handled in a single prompt. Defaults to 4096.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
         - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
         the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
         by default.
         - device (str, optional): Specifies the device for model computation ('cuda' for GPU or 'cpu'). Defaults to 'cuda'.
         - num_gpus (int, optional): Number of GPUs to use for model loading and inference. Defaults to 1.
         - variable_passages (bool, optional): Indicates whether the number of passages to rank can vary. Defaults to False.
         - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
         - system_message (Optional[str], optional): Custom system message to be included in the prompt for additional
         instructions or context. Defaults to None.

         Raises:
         - AssertionError: If CUDA is specified as the device but is not available on the system.
         - ValueError: If an unsupported prompt mode is provided.

         Note:
         - This class is operates given scenarios where listwise ranking is required, with support for dynamic
         passage handling and customization of prompts through system messages and few-shot examples.
         - GPU acceleration is supported and recommended for faster computations.
        """
        self._model = model
        self._context_size = context_size
        self._prompt_mode = prompt_mode
        self._num_few_shot_examples = num_few_shot_examples
        self._system_message = system_message
        self._variable_passages = variable_passages
        self._window_size = window_size
        self._batched = batched
        self._tokenizer = T5Tokenizer.from_pretrained("castorini/monot5-large-msmarco")
        self._output_token_estimate = 1
        self._device = device
        self._llm = T5ForConditionalGeneration.from_pretrained("castorini/monot5-large-msmarco").to(self._device)

    def run_llm_batched(
        self,
        prompts: List[str | List[Dict[str, str]]],
        current_window_size: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        if SamplingParams is None:
            raise ImportError(
                "Please install rank-llm with `pip install rank-llm[vllm]` to use batch inference."
            )
        logger.info(f"VLLM Generating!")
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.num_output_tokens(current_window_size),
            min_tokens=self.num_output_tokens(current_window_size),
        )
        outputs = self._llm.generate(prompts, sampling_params)
        print(outputs)
        print("BATCHED")

        return [
            (output.outputs[0].text, len(output.outputs[0].token_ids))
            for output in outputs
        ]

    def run_llm(
        self, prompt: str, current_window_size: Optional[int] = None
    ) -> Tuple[str, int]:
        if current_window_size is None:
            current_window_size = self._window_size
        inputs = self._tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items()}
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens(current_window_size)
        gen_cfg.min_new_tokens = self.num_output_tokens(current_window_size)
        # gen_cfg.temperature = 0
        gen_cfg.do_sample = False
        output_ids = self._llm.generate(**inputs, generation_config=gen_cfg)

        if self._llm.config.is_encoder_decoder:
            output_ids = output_ids[0]
            output_ids = output_ids[1:]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
            output_ids = output_ids[1:]

        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs, output_ids.size(0)

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        return 1

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        return f"Given the query: {query}, output its relevance to the {num} documents."

    def _add_post_prompt(self, query: str, num: int) -> str:
        return f"Given the query: {query}, output its relevance to the {num} documents."
        

    def _add_few_shot_examples(self, conv):
        return 0
        #unused for now

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        query = result.query.text
        query = self._replace_number(query)
        input = f"Query: {query} Document: {result.candidates[rank_start].doc['contents']}"
        prompt = self._tokenizer.decode(self._tokenizer.encode(input)[:480])[:-4] + " Relevant: "
        prompt = prompt.replace("<unk>","")

        return prompt, self.get_num_tokens(prompt)

    def create_prompt_batched(
        self, results: List[Result], rank_start: int, rank_end: int,
        batch_size: int = 32
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in tqdm(chunks(results, batch_size), desc="Processing batches"):
                futures = [
                    executor.submit(self.create_prompt, result, rank_start, rank_end)
                    for result in batch
                ]
                completed_prompts = [future.result() for future in as_completed(futures)]
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

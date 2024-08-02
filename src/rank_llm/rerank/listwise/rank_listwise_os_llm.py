import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastchat.model import get_conversation_template, load_model
from ftfy import fix_text
from tqdm import tqdm
from transformers.generation import GenerationConfig

from rank_llm.data import Request, Result
from rank_llm.rerank import PromptMode

from .listwise_rankllm import ListwiseRankLLM

try:
    from vllm import LLM, SamplingParams
except:
    LLM = None
    SamplingParams = None

logger = logging.getLogger(__name__)


class RankListwiseOSLLM(ListwiseRankLLM):
    def __init__(
        self,
        model: str,
        name: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
        vllm_batched: bool = False,
    ) -> None:
        """
         Creates instance of the RankListwiseOSLLM class, an extension of RankLLM designed for performing listwise ranking of passages using a specified language model. Advanced configurations are supported such as GPU acceleration, variable passage handling, and custom system messages for generating prompts.
         RankListWiseOSLLM uses the default implementations for sliding_window

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
         - vllm_batched (bool, optional): Indicates whether batched inference using VLLM is leveraged. Defaults to False.

         Raises:
         - AssertionError: If CUDA is specified as the device but is not available on the system.
         - ValueError: If an unsupported prompt mode is provided.

         Note:
         - This class is operates given scenarios where listwise ranking is required, with support for dynamic
         passage handling and customization of prompts through system messages and few-shot examples.
         - GPU acceleration is supported and recommended for faster computations.
        TODO: Make repetition_penalty configurable
        """
        super().__init__(
            model, context_size, prompt_mode, num_few_shot_examples, window_size
        )
        self._device = device
        self._vllm_batched = vllm_batched
        self._name = name
        self._variable_passages = variable_passages
        self._system_message = system_message
        self._output_token_estimate = None

        if num_few_shot_examples > 0:
            with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                self._examples = list(json_file)[1:-1]
        if self._device == "cuda":
            assert torch.cuda.is_available()

        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of {PromptMode.RANK_GPT} prompt."
            )
        if vllm_batched and LLM is None:
            raise ImportError(
                "Please install rank-llm with `pip install rank-llm[vllm]` to use batch inference."
            )
        elif vllm_batched:
            self._llm = LLM(
                model, download_dir=os.getenv("HF_HOME"), enforce_eager=False
            )
            self._tokenizer = self._llm.get_tokenizer()
        else:
            self._llm, self._tokenizer = load_model(
                model, device=device, num_gpus=num_gpus
            )

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        top_k_retrieve: int = kwargs.get("top_k_retrieve", 50)
        window_size: int = kwargs.get("window_size", 20)
        window_size = min(window_size, top_k_retrieve)
        step: int = kwargs.get("step", 10)
        populate_exec_summary: bool = kwargs.get("populate_exec_summary", False)

        if self._vllm_batched:
            # reranking using vllm
            if len(set([len(req.candidates) for req in requests])) != 1:
                raise ValueError(
                    "Batched requests must have the same number of candidates"
                )

            return self.sliding_windows_batched(
                requests,
                rank_start=max(rank_start, 0),
                rank_end=min(
                    rank_end, len(requests[0].candidates)
                ),  # TODO: Fails arbitrary hit sizes
                window_size=window_size,
                step=step,
                shuffle_candidates=shuffle_candidates,
                logging=logging,
                populate_exec_summary=populate_exec_summary,
            )
        else:
            # Normal operation mode
            results = []
            for request in tqdm(requests):
                result = self.sliding_windows(
                    request,
                    rank_start=max(rank_start, 0),
                    rank_end=min(rank_end, len(request.candidates)),
                    window_size=window_size,
                    step=step,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_exec_summary=populate_exec_summary,
                )
                results.append(result)
            return results

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
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs, output_ids.size(0)

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            _output_token_estimate = (
                len(
                    self._tokenizer.encode(
                        " > ".join([f"[{i+1}]" for i in range(current_window_size)])
                    )
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, query: str, num: int) -> str:
        example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
        return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."

    def _add_few_shot_examples(self, conv):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], response)
        return conv

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        query = result.query.text
        query = self._replace_number(query)
        num = len(result.candidates[rank_start:rank_end])
        max_length = 300 * (20 / (rank_end - rank_start))
        while True:
            conv = get_conversation_template(self._model)
            if self._system_message:
                conv.set_system_message(self._system_message)
            conv = self._add_few_shot_examples(conv)
            prefix = self._add_prefix_prompt(query, num)
            rank = 0
            input_context = f"{prefix}\n"
            for cand in result.candidates[rank_start:rank_end]:
                rank += 1
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = self.convert_doc_to_prompt_content(cand.doc, max_length)
                input_context += f"[{rank}] {self._replace_number(content)}\n"

            input_context += self._add_post_prompt(query, num)
            conv.append_message(conv.roles[0], input_context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens(
                rank_end - rank_start
            ):
                break
            else:
                max_length -= max(
                    1,
                    (
                        num_tokens
                        - self.max_tokens()
                        + self.num_output_tokens(rank_end - rank_start)
                    )
                    // ((rank_end - rank_start) * 4),
                )
        return prompt, self.get_num_tokens(prompt)

    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in tqdm(chunks(results, batch_size), desc="Processing batches"):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, rank_start, rank_end),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

    def get_name(self) -> str:
        return self._name

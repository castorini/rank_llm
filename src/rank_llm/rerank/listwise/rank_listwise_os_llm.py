import logging
import os
import random
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

import torch
import vllm
from ftfy import fix_text
from tqdm import tqdm

from rank_llm.data import Request, Result
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.vllm_handler import VllmHandler

from .listwise_rankllm import ListwiseRankLLM

try:
    import sglang
    from sglang import Engine
    from sglang.srt.entrypoints.engine import Engine as SGLangEngineType
except:
    sglang = None
    Engine = None
    SGLangEngineType = None

logger = logging.getLogger(__name__)

ALPH_START_IDX = ord("A") - 1

TEMPLATES = files("rank_llm.rerank.prompt_templates")


class RankListwiseOSLLM(ListwiseRankLLM):
    def __init__(
        self,
        model: str,
        name: str = "",
        context_size: int = 4096,
        prompt_mode: Optional[PromptMode] = None,
        prompt_template_path: Optional[str] = None,
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        stride: int = 10,
        system_message: Optional[str] = None,
        is_thinking: bool = False,
        reasoning_token_budget: int = 10000,
        use_logits: bool = False,
        use_alpha: bool = False,
        sglang_batched: bool = False,
        tensorrt_batched: bool = False,
        batch_size: int = 32,
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
         - few_shot_file (str, optional): Path to JSONL file containing few-shot examples. Required if num_few_shot_examples > 0.
         File should contain one JSON object per line with "conversations" field containing prompt/response pairs.
         - device (str, optional): Specifies the device for model computation ('cuda' for GPU or 'cpu'). Defaults to 'cuda'.
         - num_gpus (int, optional): Number of GPUs to use for model loading and inference. Defaults to 1.
         - variable_passages (bool, optional): Indicates whether the number of passages to rank can vary. Defaults to False.
         - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
         - stride (int, optional): The stride size for moving the window. Defaults to 10.
         - system_message (Optional[str], optional): Custom system message to be included in the prompt for additional
         instructions or context. Defaults to None.
         - use_logits (bool, optional): Indicates whether to use logits or not. Defaults to False.
         - use_alpha (bool, optional): Indicates whether to use alphabet ordering the prompts. Defaults to False.
         - sglang_batched (bool, optional): Indicates whether batched inference using SGLang is leveraged. Defaults to False.
         - tensorrt_batched (bool, optional): Indicates whether batched inference using TensorRT-LLM is leveraged. Defaults to False.
        - batch_size (int, optional): The size of the batch for processing requests. Defaults to 32.

         Raises:
         - AssertionError: If CUDA is specified as the device but is not available on the system.
         - ValueError: If an unsupported prompt mode is provided.
         - ValueError: If num_few_shot_examples > 0 but no valid file path is provided

         Note:
         - This class is operates given scenarios where listwise ranking is required, with support for dynamic
         passage handling and customization of prompts through system messages and few-shot examples.
         - GPU acceleration is supported and recommended for faster computations.
        TODO: Make repetition_penalty configurable
        """
        if prompt_template_path is None:
            prompt_template_path = (
                TEMPLATES / "rank_zephyr_alpha_template.yaml"
                if use_alpha
                else TEMPLATES / "rank_zephyr_template.yaml"
            )
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            window_size=window_size,
            stride=stride,
            use_alpha=use_alpha,
            device=device,
            batch_size=batch_size,
        )
        self._sglang_batched = sglang_batched
        self._tensorrt_batched = tensorrt_batched
        self._name = name
        self._variable_passages = variable_passages
        self._system_message = system_message
        self._is_thinking = is_thinking
        self._reasoning_token_budget = reasoning_token_budget
        self._output_token_estimate = None
        self._use_logits = use_logits
        self._num_gpus = num_gpus

        if self._device == "cuda":
            assert torch.cuda.is_available() and torch.cuda.device_count() >= num_gpus
        if prompt_mode and prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of {PromptMode.RANK_GPT} prompt."
            )

        if sglang_batched:
            if Engine is None:
                raise ImportError(
                    "Please install rank-llm with `pip install rank-llm[sglang]` to use sglang batch inference."
                )
            # Add assert here to ensure
            assert Engine is not None
            port = random.randint(30000, 35000)
            self._llm = Engine(model, port=port)
            self._tokenizer = self._llm.get_tokenizer()
        elif tensorrt_batched:
            try:
                from tensorrt_llm import LLM as TRTLLM
                from tensorrt_llm import BuildConfig
            except Exception:
                raise ImportError(
                    "Please install rank-llm with `pip install -e .[tensorrt-llm]` to use tensorrt batch inference."
                )
            build_config = BuildConfig(max_seq_len=4096)
            self._llm = TRTLLM(model=model, build_config=build_config)
            self._tokenizer = self._llm.tokenizer
        else:
            self._vllm_handler = VllmHandler(
                model=model,
                download_dir=os.getenv("HF_HOME"),
                enforce_eager=False,
                max_logprobs=30,
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.90,
            )
            self._tokenizer = self._vllm_handler.get_tokenizer()

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        top_k_retrieve: int = kwargs.get("top_k_retrieve", rank_end)
        rank_end = min(top_k_retrieve, rank_end)
        populate_invocations_history: bool = kwargs.get(
            "populate_invocations_history", False
        )

        # reranking using vllm or sglang
        if len(set([len(req.candidates) for req in requests])) != 1:
            raise ValueError("Batched requests must have the same number of candidates")

        return self.sliding_windows_batched(
            requests,
            rank_start=max(rank_start, 0),
            rank_end=min(
                rank_end, len(requests[0].candidates)
            ),  # TODO: Fails arbitrary hit sizes
            top_k_retrieve=top_k_retrieve,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            populate_invocations_history=populate_invocations_history,
        )

    def _evaluate_logits(
        self, logits: Dict[str, "Logit"], total: Tuple[int, int]
    ) -> Tuple[str, Dict[int, float]]:
        if self._use_alpha:
            evaluations = {
                ord(logit.decoded_token): logit.logprob
                for logit in logits.values()
                if len(logit.decoded_token) == 1
                and logit.decoded_token.isalpha()
                and ALPH_START_IDX + 1
                <= ord(logit.decoded_token)
                <= ALPH_START_IDX + self._window_size
            }
            sorted_evaluations = sorted(evaluations.items(), key=lambda x: -x[1])
            result_string = ">".join([f"[{chr(x)}]" for x, y in sorted_evaluations])
        else:
            evaluations = {
                int(logit.decoded_token): logit.logprob
                for logit in logits.values()
                if logit.decoded_token.isnumeric()
                and not unicodedata.name(logit.decoded_token).startswith(
                    ("SUPERSCRIPT", "VULGAR FRACTION", "SUBSCRIPT", "CJK UNIFIED")
                )
                and total[0] <= int(logit.decoded_token) <= total[1]
            }
            sorted_evaluations = sorted(evaluations.items(), key=lambda x: -x[1])
            result_string = ">".join([f"[{x}]" for x, y in sorted_evaluations])

        return result_string, evaluations

    def _get_logits_single_digit(
        self,
        output: vllm.RequestOutput,
        effective_location: int = 1,
        total: Tuple[int, int] = (1, 9),
    ):
        logits = output.outputs[0].logprobs[effective_location]
        return self._evaluate_logits(logits, total)

    def run_llm_batched(
        self,
        prompts: List[str | List[Dict[str, str]]],
        current_window_size: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        if current_window_size is None:
            current_window_size = self._window_size

        if hasattr(self, "_vllm_handler"):
            logger.info("VLLM Generating!")

            if self._use_logits:
                outputs = self._vllm_handler.generate_output(
                    prompts=prompts,
                    min_tokens=2,
                    max_tokens=2,
                    temperature=0.0,
                    logprobs=30,
                )
                arr = [self._get_logits_single_digit(output) for output in outputs]
                return [(s, len(s)) for s, __ in arr]
            else:
                outputs = self._vllm_handler.generate_output(
                    prompts=prompts,
                    min_tokens=self.num_output_tokens(current_window_size),
                    max_tokens=(
                        self._reasoning_token_budget
                        if self._is_thinking
                        else self.num_output_tokens(current_window_size)
                    ),
                    temperature=0.0,
                )
                return [
                    (output.outputs[0].text, len(output.outputs[0].token_ids))
                    for output in outputs
                ]
        elif (
            sglang is not None
            and SGLangEngineType is not None
            and isinstance(self._llm, SGLangEngineType)
        ):
            logger.info(f"SGLang Generating!")
            sampling_params = {
                "temperature": 0.0,
                "max_new_tokens": self.num_output_tokens(current_window_size),
                "min_new_tokens": self.num_output_tokens(current_window_size),
            }
            outputs = self._llm.generate(prompts, sampling_params)
            return [
                # completion_tokens counts stop token
                (output["text"], output["meta_info"]["completion_tokens"] - 1)
                for output in outputs
            ]
        elif self._tensorrt_batched:
            import tensorrt_llm.hlapi.llm
            from tensorrt_llm import SamplingParams as TRTSamplingParams

            if isinstance(self._llm, tensorrt_llm.hlapi.llm.LLM):
                logger.info(f"TensorRT LLM Generating!")
                sampling_params = TRTSamplingParams(
                    temperature=0.0,
                    max_tokens=self.num_output_tokens(current_window_size),
                    min_tokens=self.num_output_tokens(current_window_size),
                )
                outputs = self._llm.generate(prompts, sampling_params)
                return [
                    (output.outputs[0].text, len(output.outputs[0].token_ids))
                    for output in outputs
                ]
        else:
            raise ValueError(
                "Only support SGLang and VLLM inference backend for inferencing."
            )

    def run_llm(
        self, prompt: str, current_window_size: Optional[int] = None
    ) -> Tuple[str, int]:
        # Now forward the run_llm into run_llm_batched
        if current_window_size is None:
            current_window_size = self._window_size

        return self.run_llm_batched([prompt], current_window_size)[0]

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size

        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate

        if self._use_alpha:
            token_str = " > ".join(
                [f"[{chr(ALPH_START_IDX+i+1)}]" for i in range(current_window_size)]
            )
        else:
            token_str = " > ".join([f"[{i+1}]" for i in range(current_window_size)])

        _output_token_estimate = len(self._tokenizer.encode(token_str)) - 1

        if (
            self._output_token_estimate is None
            and self._window_size == current_window_size
        ):
            self._output_token_estimate = _output_token_estimate

        return _output_token_estimate

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        max_length = 300 * (20 / (rank_end - rank_start))

        while True:
            messages = self._inference_handler.generate_prompt(
                result=result,
                rank_start=rank_start,
                rank_end=rank_end,
                max_length=max_length,
                use_alpha=self._use_alpha,
                num_fewshot_examples=self._num_few_shot_examples,
                fewshot_examples=self._examples,
            )
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._is_thinking,
            )
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
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in tqdm(
                chunks(results, self._batch_size), desc="Processing batches"
            ):
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

import json
import logging
import os
import random
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias, Union

import numba
import torch
from fastchat.model import get_conversation_template, load_model
from ftfy import fix_text
from tqdm import tqdm
from transformers.generation import GenerationConfig

from rank_llm.data import Request, Result
from rank_llm.rerank import PromptMode

from .listwise_rankllm import ListwiseRankLLM
from .reorder.reorder_policy import ReorderPolicy

try:
    from vllm import LLM, RequestOutput, SamplingParams
except:
    LLM = None
    RequestOutput = None
    SamplingParams = None

try:
    import sglang
    import sglang.srt.server
    from sglang import Engine
except:
    sglang = None
    Engine = None

try:
    # import tensorrt_llm
    tensorrt_llm = None
except Exception as e:
    tensorrt_llm = None


logger = logging.getLogger(__name__)

ALPH_START_IDX = ord("A") - 1

# TODO Type alias for inference backend, should be changed into type alias when update to 3.12
InferenceBackend: TypeAlias = Literal["vllm", "sglang", "tensorrt", "transformers"]

ReorderMethod: TypeAlias = Literal["logits", "order"]


class RankListwiseOSLLM(ListwiseRankLLM):
    def __init__(
        self,
        model: str,
        reorder_policy: Optional[ReorderPolicy] = None,
        name: str = "",
        context_size: int = 4096,
        window_size: int = 20,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        system_message: str = None,
        use_logits: bool = False,
        use_alpha: bool = False,
        vllm_batched: bool = False,
        vllm_chunked_prefill: bool = False,
        sglang_batched: bool = False,
        tensorrt_batched: bool = False,
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
         - use_logits (bool, optional): Indicates whether to use logits or not. Defaults to False.
         - use_alpha (bool, optional): Indicates whether to use alphabet ordering the prompts. Defaults to False.
         - vllm_batched (bool, optional): Indicates whether batched inference using VLLM is leveraged. Defaults to False.
         - sglang_batched (bool, optional): Indicates whether batched inference using SGLang is leveraged. Defaults to False.
         - tensorrt_batched (bool, optional): Indicates whether batched inference using TensorRT-LLM is leveraged. Defaults to False.

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
            reorder_policy=reorder_policy,
            model=model,
            context_size=context_size,
            window_size=window_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            use_alpha=use_alpha,
        )
        self._device = device
        self._vllm_batched = vllm_batched
        self._sglang_batched = sglang_batched
        self._tensorrt_batched = tensorrt_batched
        self._name = name
        self._variable_passages = variable_passages
        self._system_message = system_message
        self._output_token_estimate = None
        self._use_logits = use_logits

        if num_few_shot_examples > 0:
            with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                self._examples = list(json_file)[1:-1]
        if self._device == "cuda":
            assert torch.cuda.is_available()

        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of {PromptMode.RANK_GPT} prompt."
            )

        # ============ Model Loading ==========

        # case of Any is the transformers load_model
        self._llm_inference_mode: InferenceBackend = None
        self._llm: Union[LLM, Engine, Any] = None

        if vllm_batched and LLM is None:
            raise ImportError(
                "Please install rank-llm with `pip install -e .[vllm]` to use batch inference."
            )
        elif vllm_batched:
            # TODO: find max_model_len given gpu
            assert LLM is not None
            self._llm_inference_mode = "vllm"
            self._llm = LLM(
                model,
                download_dir=os.getenv("HF_HOME"),
                enforce_eager=False,
                enable_chunked_prefill=vllm_chunked_prefill,
                disable_sliding_window=vllm_chunked_prefill,
                max_logprobs=30,
                tensor_parallel_size=num_gpus,
            )
            self._tokenizer = self._llm.get_tokenizer()
        elif sglang_batched and Engine is None:
            raise ImportError(
                "Please install rank-llm with `pip install -e .[sglang]` to use sglang batch inference."
            )
        elif sglang_batched:
            assert Engine is not None
            self._llm_inference_mode = "sglang"
            port = random.randint(30000, 35000)
            self._llm = Engine(model_path=model, port=port)
            self._tokenizer = self._llm.get_tokenizer()
        elif tensorrt_batched and tensorrt_llm is None:
            raise ImportError(
                "Please install rank-llm with `pip install -e .[tensorrt-llm]` to use tensorrt batch inference."
            )
        elif tensorrt_batched:
            assert tensorrt_llm is not None
            build_config = tensorrt_llm.BuildConfig(max_seq_len=4096)
            self._llm = tensorrt_llm.LLM(model=model, build_config=build_config)
            self._tokenizer = self._llm.tokenizer
        else:
            self._llm_inference_mode = "transformers"
            self._llm, self._tokenizer = load_model(
                model, device=device, num_gpus=num_gpus
            )

        # ============ Logits ==========

        self._logits_mode: ReorderMethod = "logits" if self._use_logits else "order"

        # ============ Order chars ==========

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        batched: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        return super().rerank_batch(
            requests=requests,
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            batched=(self._llm_inference_mode in ("vllm", "sglang")) or batched,
            **kwargs,
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
                    ("SUPERSCRIPT", "VULGAR FRACTION", "SUBSCRIPT")
                )
                and total[0] <= int(logit.decoded_token) <= total[1]
            }
            sorted_evaluations = sorted(evaluations.items(), key=lambda x: -x[1])
            result_string = ">".join([f"[{x}]" for x, y in sorted_evaluations])

        return result_string, evaluations

    def _get_logits_single_digit(
        self,
        output: RequestOutput,
        effective_location: int = 1,
        total: Tuple[int, int] = (1, 9),
    ):
        logits = output.outputs[0].logprobs[effective_location]
        return self._evaluate_logits(logits, total)

    def run_llm_batched(
        self,
        prompts: List[str | List[Dict[str, str]]],
        silence: bool = False,
        **kwargs,
    ) -> List[Tuple[str, int]]:
        # ensure everyone is a json string
        for prompt in prompts:
            assert isinstance(prompt, str)
        # load the json
        prompt_datas = [
            json.loads(prompt) for prompt in prompts if isinstance(prompt, str)
        ]
        prompts_s: List[str] = [prompt["prompt"] for prompt in prompt_datas]
        num_passages: List[int] = [prompt["num_passages"] for prompt in prompt_datas]
        match self._llm_inference_mode:
            case "vllm":
                if SamplingParams is None:
                    raise ImportError(
                        "Please install rank-llm with `pip install rank-llm[vllm]` to use batch inference."
                    )
                assert LLM is not None and isinstance(self._llm, LLM)

                if not self._use_logits:
                    sampling_params = [
                        SamplingParams(
                            temperature=0.0,
                            max_tokens=self.num_output_tokens(x),
                            min_tokens=0,
                        )
                        for x in num_passages
                    ]
                    outputs = self._llm.generate(
                        prompts_s, sampling_params, use_tqdm=not silence
                    )
                    return [
                        (output.outputs[0].text, len(output.outputs[0].token_ids))
                        for output in outputs
                    ]
                else:
                    sampling_params = [
                        SamplingParams(
                            temperature=0.0, max_tokens=2, min_tokens=2, logprobs=30
                        )
                        for _ in num_passages
                    ]
                    outputs = self._llm.generate(
                        prompts_s, sampling_params, use_tqdm=not silence
                    )
                    arr = [self._get_logits_single_digit(output) for output in outputs]
                    return [(s, len(s)) for s, _ in arr]
            case "sglang":
                assert sglang is not None and isinstance(
                    self._llm, sglang.srt.server.Engine
                )

                sampling_params = [
                    {
                        "temperature": 0.0,
                        "max_new_tokens": self.num_output_tokens(psg),
                        "min_new_tokens": self.num_output_tokens(psg),
                    }
                    for psg in num_passages
                ]
                outputs = self._llm.generate(prompts_s, sampling_params)
                return [
                    # completion_tokens counts stop token
                    (output["text"], output["meta_info"]["completion_tokens"] - 1)
                    for output in outputs
                ]
            case "tensorrt":
                import tensorrt_llm.hlapi.llm
                from tensorrt_llm import SamplingParams as TRTSamplingParams

                assert isinstance(self._llm, tensorrt_llm.hlapi.llm.LLM)
                sampling_params = [
                    TRTSamplingParams(
                        temperature=0.0,
                        max_tokens=self.num_output_tokens(x),
                        min_tokens=self.num_output_tokens(x),
                    )
                    for x in num_passages
                ]
                outputs = self._llm.generate(prompts_s, sampling_params)
                return [
                    (output.outputs[0].text, len(output.outputs[0].token_ids))
                    for output in outputs
                ]
            case _:
                raise ValueError(
                    f"Invalid batched inference configuration: {self._llm_inference_mode} for LLM"
                )

    def run_llm(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        silence: bool = False,
        **kwargs,
    ) -> Tuple[str, int]:
        assert isinstance(prompt, str)
        prompt_data = json.loads(prompt)
        prompt_s, num_passage_s = prompt_data["prompt"], prompt_data["num_passages"]
        num_passage = int(num_passage_s)

        if self._use_logits:
            assert (
                self._llm_inference_mode == "vllm"
                and SamplingParams is not None
                and LLM is not None
                and isinstance(self._llm, LLM)
            ), "Logits are only supported with VLLM"
            params = SamplingParams(
                min_tokens=1, max_tokens=1, temperature=0.0, logprobs=30
            )
            output = self._llm.generate(
                [prompt + "["], sampling_params=params, use_tqdm=False
            )[0]
            s, _ = self._get_logits_single_digit(output, effective_location=0)
            return s, len(s)
        else:
            inputs = self._tokenizer([prompt_s])
            inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items()}
            gen_cfg = GenerationConfig.from_model_config(self._llm.config)
            gen_cfg.max_new_tokens = self.num_output_tokens(num_passage)
            gen_cfg.min_new_tokens = self.num_output_tokens(num_passage)
            # gen_cfg.temperature = 0
            gen_cfg.do_sample = False
            output_ids = self._llm.generate(**inputs, generation_config=gen_cfg)

            if self._llm.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
            outputs = self._tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            return outputs, output_ids.size(0)

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size

        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate

        if not self._use_alpha:
            token_str = " > ".join([f"[{i+1}]" for i in range(current_window_size)])
        else:
            token_str = " > ".join(
                [f"[{chr(ALPH_START_IDX+i+1)}]" for i in range(current_window_size)]
            )

        _output_token_estimate = len(self._tokenizer.encode(token_str)) - 1

        if (
            self._output_token_estimate is None
            and self._window_size == current_window_size
        ):
            self._output_token_estimate = _output_token_estimate

        return _output_token_estimate

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        if self._use_alpha:
            return f"I will provide you with {num} passages, each indicated by a alphabetical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
        else:
            return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, query: str, num: int) -> str:
        if self._use_alpha:
            example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        else:
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

    def _add_few_shot_examples_messages(self, messages):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
        return messages

    def create_prompt(
        self, result: Result, selected_indices: List[int]
    ) -> Tuple[str, int]:
        query = result.query.text
        query = self._replace_number(query)
        num = len(selected_indices)
        max_length = int(300 * (20 / (len(selected_indices))))
        while True:
            match self._llm_inference_mode:
                case "vllm":
                    messages = list()
                    if self._system_message:
                        messages.append(
                            {"role": "system", "content": self._system_message}
                        )
                    messages = self._add_few_shot_examples_messages(messages)
                    prefix = self._add_prefix_prompt(query, num)
                    rank = 0
                    input_context = f"{prefix}\n"
                    for idx in selected_indices:
                        cand = result.candidates[idx]
                        rank += 1
                        content = self.convert_doc_to_prompt_content(
                            cand.doc, max_length
                        )

                        identifier = (
                            chr(ALPH_START_IDX + rank) if self._use_alpha else str(rank)
                        )
                        input_context += (
                            f"[{identifier}] {self._replace_number(content)}\n"
                        )

                    input_context += self._add_post_prompt(query, num)
                    messages.append({"role": "user", "content": input_context})

                    prompt = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompt = fix_text(prompt)
                    num_tokens = self.get_num_tokens(prompt)
                    if num_tokens <= self.max_tokens() - self.num_output_tokens(num):
                        break
                    else:
                        max_length -= max(
                            1,
                            (
                                num_tokens
                                - self.max_tokens()
                                + self.num_output_tokens(num)
                            )
                            // ((num) * 4),
                        )
                case _:
                    conv = get_conversation_template(self._model)
                    if self._system_message:
                        conv.set_system_message(self._system_message)
                    conv = self._add_few_shot_examples(conv)
                    prefix = self._add_prefix_prompt(query, num)
                    rank = 0
                    input_context = f"{prefix}\n"
                    for cand in selected_indices:
                        cand = result.candidates[cand]
                        rank += 1
                        # For Japanese should cut by character: content = content[:int(max_length)]
                        content = self.convert_doc_to_prompt_content(
                            cand.doc, max_length
                        )

                        identifier = (
                            chr(ALPH_START_IDX + rank) if self._use_alpha else str(rank)
                        )
                        input_context += (
                            f"[{identifier}] {self._replace_number(content)}\n"
                        )

                    input_context += self._add_post_prompt(query, num)
                    conv.append_message(conv.roles[0], input_context)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    prompt = fix_text(prompt)
                    num_tokens = self.get_num_tokens(prompt)
                    if num_tokens <= self.max_tokens() - self.num_output_tokens(num):
                        break
                    else:
                        max_length -= max(
                            1,
                            (
                                num_tokens
                                - self.max_tokens()
                                + self.num_output_tokens(num)
                            )
                            // ((num) * 4),
                        )
        return json.dumps({"prompt": prompt, "num_passages": num}), self.get_num_tokens(
            prompt
        )

    def create_prompt_batched(
        self,
        results: List[Result],
        selected_indices_batch: List[List[int]],
        batch_size: int = 32,
    ) -> List[Tuple[Union[str, List[Dict[str, str]]], int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in tqdm(
                chunks(list(zip(results, selected_indices_batch)), batch_size),
                desc="Processing batches",
                leave=False,
            ):
                completed_prompts = list(
                    executor.map(lambda req: self.create_prompt(req[0], req[1]), batch)
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        assert isinstance(prompt, str), "Prompt should be a string in RankListwiseOSLLM"
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

    def get_name(self) -> str:
        return self._name

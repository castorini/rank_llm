from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import T5Tokenizer

from rank_llm.data import Request, Result
from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM
from rank_llm.rerank.listwise.lit5.model import FiD, FiDCrossAttentionScore
from rank_llm.rerank.rankllm import PromptMode


class RankFiDDistill(ListwiseRankLLM):
    def _post_init(self):
        self._to_precision(self._precision)

    def _tokenize(self, s: str):
        return self._tokenizer(s)

    def _to_precision(self, precision: str) -> None:
        """
        We don't support python12 for now, after python 12, the code should be changed into
        """
        if precision == "float32":
            self._llm = self._llm.float()
        elif precision == "bfloat16":
            self._llm = self._llm.bfloat16()
        elif precision == "float16":
            self._llm = self._llm.float16()

    def __init__(
        self,
        model: str,
        context_size: int = 150,
        prompt_mode: PromptMode = PromptMode.LiT5,  # Placeholder for actual mode
        prompt_template_path: str = "src/rank_llm/rerank/prompt_templates/rank_fid_template.yaml",
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        window_size: int = 20,
        stride: int = 10,
        precision: str = "bfloat16",
        device: str = "cuda",
    ) -> None:
        """
        Creates instance of the RankFiDDistill class, a specialized version of RankLLM designed from Lit5-Distill.
        """
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            window_size=window_size,
        )
        self._precision = precision
        self._tokenizer = T5Tokenizer.from_pretrained(model)
        self._llm = FiD.from_pretrained(model).to(device).eval()

        self._device = device

        self._window_size = window_size
        self._stride = stride

        self._answer_maxlength = len(
            " > ".join(map(lambda x: f"[{x}]", range(1, window_size + 1)))
        )

        self._output_token_estimate = None

        self._post_init()

    def _run_llm_by_length_unified(
        self, batch_prompts: List[List[str]]
    ) -> List[Tuple[str, int]]:
        if len(batch_prompts) == 0:
            return []

        self._llm.eval()

        batch_size = len(batch_prompts)
        n_passages = len(batch_prompts[0])

        # single batch, unsqueeze
        inputs = {
            k: v.reshape(batch_size, -1).to(self._device)
            for k, v in self._tokenizer(
                [prompt for prompts in batch_prompts for prompt in prompts],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_tokens(),
            ).items()
        }

        with torch.no_grad():
            self._llm.reset_n_passages(n_passages=n_passages)
            outputs = self._llm.generate(
                **inputs,
                max_length=self._answer_maxlength,
                do_sample=False,
            )

        decoded_outputs = [
            self._tokenizer.decode(outputs[i], skip_special_tokens=True)
            for i in range(outputs.shape[0])
        ]

        # all token size should be equal
        return [
            (decoded_output, outputs.shape[1]) for decoded_output in decoded_outputs
        ]

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
        window_size: int = kwargs.get("window_size", self._window_size)
        window_size = min(window_size, top_k_retrieve)
        stride: int = kwargs.get("stride", self._stride)
        populate_invocations_history: bool = kwargs.get(
            "populate_invocations_history", False
        )
        batch_size = kwargs.get("batch_size", 1)

        if len(set([len(req.candidates) for req in requests])) != 1:
            raise ValueError("Batched requests must have the same number of candidates")

        result = []

        with tqdm(range(0, len(requests))) as bar:
            for i in range(0, len(requests), batch_size):
                batch = requests[i : min(i + batch_size, len(requests))]
                batch_result = self.sliding_windows_batched(
                    batch,
                    rank_start=max(rank_start, 0),
                    rank_end=min(
                        rank_end, len(requests[0].candidates)
                    ),  # TODO: Fails arbitrary hit sizes
                    window_size=window_size,
                    stride=stride,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_invocations_history=populate_invocations_history,
                )
                result.extend(batch_result)
                bar.update(len(batch))

        return result

    def run_llm_batched(
        self, prompts: List[List[Dict[str, str]]], **kwargs
    ) -> List[Tuple[str, int]]:
        if len(prompts) == 0:
            return []

        # unfortunately, we are not allowed to use VLLM on T5. However, we could unify the prompts by passage size
        #   (which is commonly the same) then rerank stuff having same passage sizes

        prompt_infos = [list(map(lambda x: x["text"], prompt)) for prompt in prompts]

        return self._run_llm_by_length_unified(prompt_infos)

    def create_prompt_batched(
        self, results: List[Result], rank_start: int, rank_end: int, batch_size: int
    ) -> List[Tuple[List[Dict[str, str]], int]]:
        return [self.create_prompt(result, rank_start, rank_end) for result in results]

    def run_llm(self, prompts: List[Dict[str, str]], **kwargs) -> Tuple[str, int]:
        """
        Run the target language model with a passed in prompt.
        """

        return self._run_llm_by_length_unified(
            [list(map(lambda x: x["text"], prompts))]
        )[0]

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Create a prompt based on the result and given ranking range.
        """

        max_length = 300 * (20 / (rank_end - rank_start))
        messages = self._inference_handler.generate_prompt(
            result=result,
            rank_start=rank_start,
            rank_end=rank_end,
            max_length=max_length,
        )

        # For now, we concat the prompt, because it seems LiT5 is also concatting the stuff
        prompts = [
            {"text": messages[i]} for i in range(0, 2 * (rank_end - rank_start), 2)
        ]

        return prompts, sum(self.get_num_tokens(prompt["text"]) for prompt in prompts)

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """
        Abstract method to calculate the number of tokens contained in the given prompt.
        """
        if isinstance(prompt, str):
            return len(self._tokenizer.encode(prompt))
        elif isinstance(prompt, list):
            return sum(len(self._tokenizer.encode(item["text"])) for item in prompt)
        else:
            raise ValueError(
                "Prompt must be a string or a list of dictionaries with a 'text' key."
            )

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if (
            self._output_token_estimate is not None
            and self._window_size == current_window_size
        ):
            return self._output_token_estimate
        else:
            output_token_estimate = (
                len(
                    self._tokenizer.encode(
                        " > ".join([f"[{i + 1}]" for i in range(current_window_size)])
                    )
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = output_token_estimate

            return output_token_estimate


class RankFiDScore(ListwiseRankLLM):
    def _post_init(self):
        # set the overwrite forward cross attention
        self._llm.overwrite_forward_crossattention()
        self._to_precision(self._precision)

    def _tokenize(self, s: str):
        return self._tokenizer(s)

    def _to_precision(self, precision: str) -> None:
        """
        We don't support python12 for now, after python 12, the code should be changed into
        """
        if precision == "float32":
            self._llm = self._llm.float()
        elif precision == "bfloat16":
            self._llm = self._llm.bfloat16()
        elif precision == "float16":
            self._llm = self._llm.float16()

    def __init__(
        self,
        model: str,
        context_size: int = 150,
        prompt_mode: PromptMode = PromptMode.LiT5,  # Placeholder for actual mode
        prompt_template_path: str = "src/rank_llm/rerank/prompt_templates/rank_fid_score_template.yaml",
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        window_size: int = 20,
        stride: int = 10,
        precision: str = "bfloat16",
        device: str = "cuda",
    ) -> None:
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            window_size=window_size,
        )
        self._precision = precision
        self._tokenizer = T5Tokenizer.from_pretrained(model)
        self._llm = FiDCrossAttentionScore.from_pretrained(model).to(device).eval()

        self._device = device
        self._window_size = window_size
        self._stride = stride

        self._output_token_estimate = None

        self._post_init()

    def _run_llm_by_length_unified(
        self, batch_prompts: List[List[Tuple[str, str]]]
    ) -> List[Tuple[str, int]]:
        if len(batch_prompts) == 0:
            return []

        # get arbitrary query (they should be the same)
        queries = [prompts[0][0] for prompts in batch_prompts]
        batch_size = len(batch_prompts)
        n_passages = len(batch_prompts[0])

        inputs = {
            k: v.reshape(batch_size, -1).to(self._device)
            for k, v in self._tokenizer(
                [prompt for prompts in batch_prompts for (_, prompt) in prompts],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_tokens(),
            ).items()
        }

        passage_ids = inputs["input_ids"]
        passage_mask = inputs["attention_mask"]

        with torch.no_grad():
            self._llm.reset_score_storage()

            outputs = self._llm.generate(
                **inputs, max_length=20, do_sample=False, n_passages=n_passages
            )

        output_sequence_lengths = []

        for output in outputs:
            output_length = 0
            for j in range(output.shape[0]):
                if output[j] == FiDCrossAttentionScore.ANSWER_EOS_TOKEN:
                    output_length = j
                    break
            else:
                output_length = outputs.shape[1]
            output_sequence_lengths.append(output_length)

        query_mask_reader = self._tokenizer(
            queries,
            max_length=self.max_tokens(),
            padding="longest",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )["attention_mask"].bool()

        with torch.no_grad():
            crossattention_scores = self._llm.get_crossattention_scores(
                n_passages,
                ids=passage_ids.to(self._device),
                mask=passage_mask.bool().to(self._device),
                mask_query=query_mask_reader.to(self._device),
                output_sequence_lengths=output_sequence_lengths,
            )
            # only supports normswoquery for now
            crossattention_score: torch.Tensor = crossattention_scores["normswoquery"]
            sorted, idxes = torch.sort(crossattention_score, dim=-1, descending=True)
            idxes = idxes.detach().cpu()

        return [
            (
                " > ".join([f"[{x + 1}]" for x in idxes[i].tolist()]),
                output_sequence_lengths[i] + crossattention_score.shape[1],
            )
            for i in range(idxes.shape[0])
        ]

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
        window_size: int = kwargs.get("window_size", self._window_size)
        window_size = min(window_size, top_k_retrieve)
        stride: int = kwargs.get("stride", self._stride)
        populate_invocations_history: bool = kwargs.get(
            "populate_invocations_history", False
        )
        batch_size = kwargs.get("batch_size", 1)

        if len(set([len(req.candidates) for req in requests])) != 1:
            raise ValueError("Batched requests must have the same number of candidates")

        result = []

        with tqdm(range(0, len(requests))) as bar:
            for i in range(0, len(requests), batch_size):
                batch = requests[i : min(i + batch_size, len(requests))]
                batch_result = self.sliding_windows_batched(
                    batch,
                    rank_start=max(rank_start, 0),
                    rank_end=min(
                        rank_end, len(requests[0].candidates)
                    ),  # TODO: Fails arbitrary hit sizes
                    window_size=window_size,
                    stride=stride,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_invocations_history=populate_invocations_history,
                )
                result.extend(batch_result)
                bar.update(len(batch))

        return result

    def run_llm_batched(
        self, prompts: List[List[Dict[str, str]]], **kwargs
    ) -> List[Tuple[str, int]]:
        if len(prompts) == 0:
            return []

        # unfortunately, we are not allowed to use VLLM on T5. However, we could unify the prompts by passage size
        #   (which is commonly the same) then rerank stuff having same passage sizes

        processed_prompts = [
            [(x["query"], x["text"]) for x in prmpt] for prmpt in prompts
        ]

        return self._run_llm_by_length_unified(processed_prompts)

    def create_prompt_batched(
        self, results: List[Result], rank_start: int, rank_end: int, batch_size: int
    ) -> List[Tuple[List[Dict[str, str]], int]]:
        return [self.create_prompt(result, rank_start, rank_end) for result in results]

    def run_llm(self, prompts: List[Dict[str, str]], **kwargs) -> Tuple[str, int]:
        # get arbitrary query (they should be the same)
        return self._run_llm_by_length_unified(
            [[(x["query"], x["text"]) for x in prompts]]
        )[0]

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Create a prompt based on the result and given ranking range.
        """
        query = result.query.text
        results = []

        sum_token = 0
        max_length = 300 * (20 / (rank_end - rank_start))

        # TODO (issue #237): Need to modify this to add fewshot examples
        messages = self._inference_handler.generate_prompt(
            result=result,
            rank_start=rank_start,
            rank_end=rank_end,
            max_length=max_length,
        )
        query_part = messages[0]["content"]
        body_messages = messages[2:]

        for i in range(0, 2 * (rank_end - rank_start), 2):
            results.append({"query": query_part, "text": body_messages[i]})
            sum_token += len(self._tokenizer.encode(results[-1]["text"]))

        return results, sum_token

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0.0

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if (
            self._output_token_estimate is not None
            and self._window_size == current_window_size
        ):
            return self._output_token_estimate
        else:
            output_token_estimate = (
                len(
                    self._tokenizer.encode(
                        " > ".join([f"[{i + 1}]" for i in range(current_window_size)])
                    )
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = output_token_estimate

            return output_token_estimate

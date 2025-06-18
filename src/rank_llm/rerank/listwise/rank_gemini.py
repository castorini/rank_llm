import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from rank_llm.data import Request, Result
from rank_llm.rerank import PromptMode

from .listwise_rankllm import ListwiseRankLLM

"""
make sure to run
pip install -U -q "google-generativeai>=0.8.2"
"""
try:
    import google.generativeai as genai
except:
    genai = None


def populate_generation_config(**kwargs) -> Dict[str, Any]:
    # TODO: complete this for the rest of the optional generation params.
    generation_config = {"response_mime_type": "text/plain"}
    if "temperature" in kwargs:
        generation_config["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        generation_config["top_p"] = kwargs["top_p"]
    if "top_k" in kwargs:
        generation_config["top_k"] = kwargs["top_k"]
    if "max_output_tokens" in kwargs:
        generation_config["max_output_tokens"] = kwargs["max_output_tokens"]
    return generation_config


class SafeGenai(ListwiseRankLLM):
    # TODO switch to the new genai api?
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        prompt_template_path: Optional[str] = None,
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        window_size: int = 20,
        keys=None,
        key_start_id=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            window_size=window_size,
        )
        if not genai:
            raise ImportError(
                'Please install genai with `pip install -U -q "google-generativeai>=0.8.2"` to use gemini.'
            )
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("Please provide Genai API Keys.")
        if prompt_mode not in [PromptMode.RANK_GPT_APEER, PromptMode.RANK_GPT]:
            raise ValueError(
                f"unsupported prompt mode for GEMINI models: {prompt_mode}, expected {PromptMode.RANK_GPT_APEER} or {PromptMode.RANK_GPT}."
            )
        self._output_token_estimate = None
        self._keys = keys
        self._cur_key_id = key_start_id or 0
        self._cur_key_id = self._cur_key_id % len(self._keys)
        self.generation_config = populate_generation_config(**kwargs)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        ]
        self.system_instruction = kwargs.get(
            "system_instruction",
            "As RankGemini, your task is to evaluate and rank unique passages based on their relevance and accuracy to a given query. Prioritize passages that directly address the query and provide detailed, correct answers. Ignore factors such as length, complexity, or writing style unless they seriously hinder readability.",
        )
        self.model = genai.GenerativeModel(
            model_name=self._model,
            generation_config=self.generation_config,
            system_instruction=self.system_instruction,
            safety_settings=self.safety_settings,
        )
        genai.configure(api_key=self._keys[self._cur_key_id])

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
        window_size: int = kwargs.get("window_size", 20)
        window_size = min(window_size, top_k_retrieve)
        stride: int = kwargs.get("stride", 10)
        populate_invocations_history: bool = kwargs.get(
            "populate_invocations_history", False
        )
        results = []
        for request in tqdm(requests):
            result = self.sliding_windows(
                request,
                rank_start=max(rank_start, 0),
                rank_end=min(rank_end, len(request.candidates)),
                window_size=window_size,
                stride=stride,
                shuffle_candidates=shuffle_candidates,
                logging=logging,
                populate_invocations_history=populate_invocations_history,
            )
            results.append(result)
        return results

    def run_llm_batched(self):
        pass

    def _call_inference(
        self, messages, return_text=False
    ) -> Union[str, Dict[str, Any]]:
        while True:
            try:
                if isinstance(messages, list):
                    history = messages[:-1]
                    chat_message = messages[-1]
                    chat_session = self.model.start_chat(history=history)
                    completion = chat_session.send_message(chat_message)
                else:
                    chat_session = self.model.start_chat(history=[])
                    completion = chat_session.send_message(messages)
                break
            except Exception as e:
                print("Error in completion call")
                print(str(e))
                # TODO: do not retry for some of the deterministic failures.
                self._cur_key_id = (self._cur_key_id + 1) % len(self._keys)
                genai.api_key = self._keys[self._cur_key_id]
                time.sleep(1.0)

        if return_text:
            return completion.text
        return completion

    def run_llm(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        current_window_size: Optional[int] = None,
    ) -> Tuple[str, int]:
        response = self._call_inference(
            messages=prompt,
            return_text=True,
        )
        return response, self.model.count_tokens(response).total_tokens

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        if self._prompt_mode == PromptMode.RANK_GPT_APEER:
            return f"In response to the query: [querystart] {query} [queryend], rank the passages. Ignore aspects like length, complexity, or writing style, and concentrate on passages that provide a comprehensive understanding of the query. Take into account any inaccuracies or vagueness in the passages when determining their relevance."
        return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, query: str, num: int) -> str:
        if self._prompt_mode == PromptMode.RANK_GPT_APEER:
            return f"\nGiven the query: [querystart] {query} [queryend], produce a succinct and clear ranking of all passages, from most to least relevant, using their identifiers. The format should be [rankstart] [most relevant passage ID] > [next most relevant passage ID] > ... > [least relevant passage ID] [rankend], e.g., [rankstart] [1] > ... > [2] [rankend]. Refrain from including any additional commentary or explanations in your ranking."
        return f"\nSearch Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [2] > [1], Only respond with the ranking results, do not say any word or explain."

    def _add_few_shot_examples_text(self, text_prompt: str) -> str:
        """Adds few-shot examples to a text prompt, returning the combined string."""

        if (
            self._num_few_shot_examples > 0
            and hasattr(self, "_examples")
            and self._examples
        ):
            examples_text = []
            for _ in range(min(self._num_few_shot_examples, len(self._examples))):
                ex = random.choice(self._examples)

                if "conversations" in ex and len(ex) >= 2:
                    examples_text.append(
                        f"Example Input:\n{ex['conversations'][0]['value'].strip()}\n"
                        f"Example Output:\n{ex['conversations'][1]['value'].strip()}"
                    )

            final_text = (
                "In response to the query: [querystart] {QUERY} [queryend], rank the passages. Ignore aspects like length, complexity, or writing style, and concentrate on passages that provide a comprehensive understanding of the query. Take into account any inaccuracies or vagueness in the passages when determining their relevance."
                + f"Examples:\n"
                + "\n\n".join(examples_text)
            )
            return final_text if examples_text else text_prompt
        else:
            return text_prompt

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])
        max_length = 300 * (self._window_size / (rank_end - rank_start))
        while True:
            message = self._add_few_shot_examples_text("")
            message += self._add_prefix_prompt(query, num)

            rank = 0
            for cand in result.candidates[rank_start:rank_end]:
                rank += 1
                content = self.convert_doc_to_prompt_content(cand.doc, max_length)
                message += f"\n[{rank}] {self._replace_number(content)}"
            message += self._add_post_prompt(query, num)
            num_tokens = self.get_num_tokens(message)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // ((rank_end - rank_start) * 4),
                )
        return message, self.get_num_tokens(message)

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            _output_token_estimate = (
                self.model.count_tokens(
                    " > ".join([f"[{i+1}]" for i in range(current_window_size)])
                ).total_tokens
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate

    def create_prompt_batched(
        self, results: List[Result], rank_start: int, rank_end: int, batch_size: int
    ) -> List[Tuple[List[Dict[str, str]], int]]:
        return [self.create_prompt(result, rank_start, rank_end) for result in results]

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """Returns the number of tokens used by a list of messages in prompt."""
        num_tokens = 0
        if isinstance(prompt, list):
            for message in prompt:
                for key, value in message.items():
                    response = self.model.count_tokens(value).total_tokens
                    num_tokens += response
        else:
            response = self.model.count_tokens(prompt).total_tokens
            num_tokens += response
        num_tokens += 3
        return num_tokens

    def cost_per_1k_token(self, input_token: bool) -> float:
        # TODO: add proper costs
        return 0

    def get_name(self) -> str:
        return self._model

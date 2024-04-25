import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
import tiktoken

from rank_llm.rerank.rankllm import PromptMode, RankLLM
from rank_llm.data import Result


class SafeOpenai(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        window_size: int = 20,
        keys=None,
        key_start_id=None,
        proxy=None,
        api_type: str = None,
        api_base: str = None,
        api_version: str = None,
    ) -> None:
        """
        Creates instance of the SafeOpenai class, a specialized version of RankLLM designed for safely handling OpenAI API calls with
        support for key cycling, proxy configuration, and Azure AI conditional integration.

        Parameters:
        - model (str): The model identifier for the LLM (model identifier information can be found via OpenAI's model lists).
        - context_size (int): The maximum number of tokens that the model can handle in a single request.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
        - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
        the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
        by default.
        - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
        - keys (Union[List[str], str], optional): A list of OpenAI API keys or a single OpenAI API key.
        - key_start_id (int, optional): The starting index for the OpenAI API key cycle.
        - proxy (str, optional): The proxy configuration for OpenAI API calls.
        - api_type (str, optional): The type of API service, if using Azure AI as the backend.
        - api_base (str, optional): The base URL for the API, applicable when using Azure AI.
        - api_version (str, optional): The API version, necessary for Azure AI integration.

        Raises:
        - ValueError: If an unsupported prompt mode is provided or if no OpenAI API keys / invalid OpenAI API keys are supplied.

        Note:
        - This class supports cycling between multiple OpenAI API keys to distribute quota usage or handle rate limiting.
        - Azure AI integration is depends on the presence of `api_type`, `api_base`, and `api_version`.
        """
        super().__init__(model, context_size, prompt_mode, num_few_shot_examples)
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("Please provide OpenAI Keys.")
        if prompt_mode not in [PromptMode.RANK_GPT, PromptMode.LRL]:
            raise ValueError(
                f"unsupported prompt mode for GPT models: {prompt_mode}, expected {PromptMode.RANK_GPT} or {PromptMode.LRL}."
            )

        self._window_size = window_size
        self._output_token_estimate = None
        self._keys = keys
        self._cur_key_id = key_start_id or 0
        self._cur_key_id = self._cur_key_id % len(self._keys)
        openai.proxy = proxy
        openai.api_key = self._keys[self._cur_key_id]
        self.use_azure_ai = False

        if all([api_type, api_base, api_version]):
            # See https://learn.microsoft.com/en-US/azure/ai-services/openai/reference for list of supported versions
            openai.api_version = api_version
            openai.api_type = api_type
            openai.api_base = api_base
            self.use_azure_ai = True

    class CompletionMode(Enum):
        UNSPECIFIED = 0
        CHAT = 1
        TEXT = 2

    def _call_completion(
        self,
        *args,
        completion_mode: CompletionMode,
        return_text=False,
        reduce_length=False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        while True:
            try:
                if completion_mode == self.CompletionMode.CHAT:
                    completion = openai.chat.completions.create(
                        *args, **kwargs, timeout=30
                    )
                elif completion_mode == self.CompletionMode.TEXT:
                    completion = openai.Completion.create(*args, **kwargs)
                else:
                    raise ValueError(
                        "Unsupported completion mode: %V" % completion_mode
                    )
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print("reduce_length")
                    return "ERROR::reduce_length"
                if "The response was filtered" in str(e):
                    print("The response was filtered")
                    return "ERROR::The response was filtered"
                self._cur_key_id = (self._cur_key_id + 1) % len(self._keys)
                openai.api_key = self._keys[self._cur_key_id]
                time.sleep(0.1)
        if return_text:
            completion = (
                completion.choices[0].message.content
                if completion_mode == self.CompletionMode.CHAT
                else completion.choices[0].text
            )
        return completion

    def run_llm(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        current_window_size: Optional[int] = None,
    ) -> Tuple[str, int]:
        model_key = "engine" if self.use_azure_ai else "model"
        response = self._call_completion(
            messages=prompt,
            temperature=0,
            completion_mode=SafeOpenai.CompletionMode.CHAT,
            return_text=True,
            **{model_key: self._model},
        )
        try:
            encoding = tiktoken.get_encoding(self._model)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        return response, len(encoding.encode(response))

    def _get_prefix_for_rank_gpt_prompt(
        self, query: str, num: int
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
            },
            {
                "role": "user",
                "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

    def _get_suffix_for_rank_gpt_prompt(self, query: str, num: int) -> str:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            try:
                encoder = tiktoken.get_encoding(self._model)
            except:
                encoder = tiktoken.get_encoding("cl100k_base")

            _output_token_estimate = (
                len(
                    encoder.encode(
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

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[List[Dict[str, str]], int]:
        if self._prompt_mode == PromptMode.RANK_GPT:
            return self.create_rank_gpt_prompt(result, rank_start, rank_end)
        else:
            return self.create_LRL_prompt(result, rank_start, rank_end)

    def create_rank_gpt_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[List[Dict[str, str]], int]:
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])

        max_length = 300 * (self._window_size / (rank_end - rank_start))
        while True:
            messages = self._get_prefix_for_rank_gpt_prompt(query, num)
            rank = 0
            for cand in result.candidates[rank_start:rank_end]:
                rank += 1
                content = self.covert_doc_to_prompt_content(cand.doc, max_length)
                messages.append(
                    {
                        "role": "user",
                        "content": f"[{rank}] {self._replace_number(content)}",
                    }
                )
                messages.append(
                    {"role": "assistant", "content": f"Received passage [{rank}]."}
                )
            messages.append(
                {
                    "role": "user",
                    "content": self._get_suffix_for_rank_gpt_prompt(query, num),
                }
            )
            num_tokens = self.get_num_tokens(messages)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // ((rank_end - rank_start) * 4),
                )
        return messages, self.get_num_tokens(messages)

    def create_LRL_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[List[Dict[str, str]], int]:
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])
        max_length = 300 * (20 / (rank_end - rank_start))
        psg_ids = []
        while True:
            message = "Sort the list PASSAGES by how good each text answers the QUESTION (in descending order of relevancy).\n"
            rank = 0
            for cand in result.candidates[rank_start:rank_end]:
                rank += 1
                psg_id = f"PASSAGE{rank}"
                content = self.covert_doc_to_prompt_content(cand.doc, max_length)
                message += f'{psg_id} = "{self._replace_number(content)}"\n'
                psg_ids.append(psg_id)
            message += f'QUESTION = "{query}"\n'
            message += "PASSAGES = [" + ", ".join(psg_ids) + "]\n"
            message += "SORTED_PASSAGES = [\n"
            messages = [{"role": "user", "content": message}]
            num_tokens = self.get_num_tokens(messages)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // ((rank_end - rank_start) * 4),
                )
        return messages, self.get_num_tokens(messages)

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """Returns the number of tokens used by a list of messages in prompt."""
        if self._model in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo"]:
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif self._model in ["gpt-4-0314", "gpt-4"]:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message, tokens_per_name = 0, 0

        try:
            encoding = tiktoken.get_encoding(self._model)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        if isinstance(prompt, list):
            for message in prompt:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        else:
            num_tokens += len(encoding.encode(prompt))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def cost_per_1k_token(self, input_token: bool) -> float:
        # Brought in from https://openai.com/pricing on 2023-07-30
        cost_dict = {
            ("gpt-3.5", 4096): 0.0015 if input_token else 0.002,
            ("gpt-3.5", 16384): 0.003 if input_token else 0.004,
            ("gpt-4", 8192): 0.03 if input_token else 0.06,
            ("gpt-4", 32768): 0.06 if input_token else 0.12,
        }
        model_key = "gpt-3.5" if "gpt-3" in self._model else "gpt-4"
        return cost_dict[(model_key, self._context_size)]

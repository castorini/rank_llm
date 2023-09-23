import copy
from enum import Enum
from pathlib import Path
import json
import time
import openai
import tiktoken
from rank_llm import RankLLM, PromptMode
from ftfy import fix_text
import re


def replace_number(s):
    return re.sub(r"\[(\d+)\]", r"(\1)", s)


class SafeOpenai(RankLLM):
    def __init__(
        self,
        model,
        context_size,
        top_k_candidates,
        dataset,
        prompt_mode,
        keys=None,
        key_start_id=None,
        proxy=None,
    ):
        super().__init__(model, context_size, top_k_candidates, dataset, prompt_mode)
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise "Please provide OpenAI Keys."
        if prompt_mode not in [PromptMode.RANK_GPT, PromptMode.LRL]:
            raise ValueError(
                "unsupported prompt mode for GPT models: {prompt_mode}, expected RANK_GPT or LRL."
            )

        self.keys = keys
        self.cur_key_id = key_start_id or 0
        self.cur_key_id = self.cur_key_id % len(self.keys)
        openai.proxy = proxy
        openai.api_key = self.keys[self.cur_key_id]

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
    ):
        while True:
            try:
                if completion_mode == self.CompletionMode.CHAT:
                    completion = openai.ChatCompletion.create(
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
                self.cur_key_id = (self.cur_key_id + 1) % len(self.keys)
                openai.api_key = self.keys[self.cur_key_id]
                time.sleep(0.1)
        if return_text:
            completion = (
                completion["choices"][0]["message"]["content"]
                if completion_mode == self.CompletionMode.CHAT
                else completion["choices"][0]["text"]
            )
        return completion

    def run_llm(self, messages):
        response = self._call_completion(
            model=self.model_,
            messages=messages,
            temperature = 0,
            completion_mode=SafeOpenai.CompletionMode.CHAT,
            return_text=True,
        )
        try:
            encoding = tiktoken.get_encoding(self.model_)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        return response, len(encoding.encode(response))

    def _get_prefix_for_rank_gpt_prompt(self, query, num):
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

    def _get_suffix_for_rank_gpt_prompt(self, query, num):
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

    def num_output_tokens(self):
        return 200

    def create_prompt(self, retrieved_result, rank_start, rank_end):
        if self.prompt_mode_ == PromptMode.RANK_GPT:
            return self.create_rank_gpt_prompt(retrieved_result, rank_start, rank_end)
        else:
            return self.create_LRL_prompt(retrieved_result, rank_start, rank_end)

    def create_rank_gpt_prompt(self, retrieved_result, rank_start, rank_end):
        query = retrieved_result["query"]
        num = len(retrieved_result["hits"][rank_start:rank_end])

        max_length = 300
        while True:
            messages = self._get_prefix_for_rank_gpt_prompt(query, num)
            rank = 0
            for hit in retrieved_result["hits"][rank_start:rank_end]:
                rank += 1
                content = hit["content"]
                content = content.replace("Title: Content: ", "")
                content = content.strip()
                content = fix_text(content)
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = " ".join(content.split()[: int(max_length)])
                messages.append(
                    {"role": "user", "content": f"[{rank}] {replace_number(content)}"}
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
                    // (rank_end - rank_start),
                )
        return messages, self.get_num_tokens(messages)

    def create_LRL_prompt(self, retrieved_result, rank_start, rank_end):
        query = retrieved_result["query"]
        num = len(retrieved_result["hits"][rank_start:rank_end])

        max_length = 300
        psg_ids = []
        while True:
            message = "Sort the list PASSAGES by how good each text answers the QUESTION (in descending order of relevancy).\n"
            rank = 0
            for hit in retrieved_result["hits"][rank_start:rank_end]:
                rank += 1
                psg_id = f"PASSAGE{rank}"
                content = hit["content"]
                content = content.replace("Title: Content: ", "")
                content = content.strip()
                content = fix_text(content)
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = " ".join(content.split()[: int(max_length)])
                message += f'{psg_id} = "{replace_number(content)}"\n'
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
                    // (rank_end - rank_start),
                )
        return messages, self.get_num_tokens(messages)

    def get_num_tokens(self, messages):
        """Returns the number of tokens used by a list of messages."""
        if self.model_ in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo"]:
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif self.model_ in ["gpt-4-0314", "gpt-4"]:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message, tokens_per_name = 0, 0

        try:
            encoding = tiktoken.get_encoding(self.model_)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        if isinstance(messages, list):
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        else:
            num_tokens += len(encoding.encode(messages))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def cost_per_1k_token(self, input_token: bool):
        # Brought in from https://openai.com/pricing on 2023-07-30
        cost_dict = {
            ("gpt-3.5", 4096): 0.0015 if input_token else 0.002,
            ("gpt-3.5", 16384): 0.003 if input_token else 0.004,
            ("gpt-4", 8192): 0.03 if input_token else 0.06,
            ("gpt-4", 32768): 0.06 if input_token else 0.12,
        }
        model_key = "gpt-3.5" if "gpt-3" in self.model_ else "gpt-4"
        return cost_dict[(model_key, self.context_size_)]

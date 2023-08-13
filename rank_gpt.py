import copy
from enum import Enum
from pathlib import Path
import json
import time
import openai
import tiktoken
from tqdm import tqdm
from pyserini_retriever import PyseriniRetriever
from rank_llm import RankLLM, PromptMode
from topics_dict import TOPICS


class SafeOpenai(RankLLM):
    def __init__(self, model, context_size, dataset, prompt_mode, keys=None, key_start_id=None, proxy=None):
        super().__init__(model, context_size, dataset, prompt_mode)
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise "Please provide OpenAI Keys."

        self.keys = keys
        self.cur_key_id = key_start_id or 0
        self.cur_key_id = self.cur_key_id % len(self.keys)
        openai.proxy = proxy
        openai.api_key = self.keys[self.cur_key_id]
    
    class CompletionMode(Enum):
        UNSPECIFIED = 0
        CHAT = 1
        TEXT = 2

    def _call_completion(self, *args, completion_mode:CompletionMode, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                if completion_mode == self.CompletionMode.CHAT:
                    completion = openai.ChatCompletion.create(*args, **kwargs, timeout=30)
                elif completion_mode == self.CompletionMode.TEXT:
                    completion = openai.Completion.create(*args, **kwargs)
                else:
                    raise ValueError("Unsupported completion mode: %V" % completion_mode)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                self.cur_key_id = (self.cur_key_id + 1) % len(self.keys)
                openai.api_key = self.keys[self.cur_key_id]
                time.sleep(0.1)
        if return_text:
            completion = (completion['choices'][0]['message']['content'] 
                          if completion_mode == self.CompletionMode.CHAT
                          else completion['choices'][0]['text'])
        return completion
    
    def run_llm(self, messages):
        response = self._call_completion(model=self.model_, messages=messages, temperature=0,
                completion_mode=SafeOpenai.CompletionMode.CHAT, return_text=True)
        try:
            encoding = tiktoken.get_encoding(self.model_)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        return response, len(encoding.encode(response))
    
    def _get_prefix_prompt(self, query, num):
        return [{'role': 'system',
                'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
                {'role': 'user',
                'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
                {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

    def _get_post_prompt(self, query, num):
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."

    def num_output_tokens(self):
        return 200
    
    def create_prompt(self, retrieved_result, rank_start=0, rank_end=100):
        query = retrieved_result['query']
        num = len(retrieved_result['hits'][rank_start: rank_end])

        max_length = 300
        while True:
            messages = self._get_prefix_prompt(query, num)
            rank = 0
            for hit in retrieved_result['hits'][rank_start: rank_end]:
                rank += 1
                content = hit['content']
                content = content.replace('Title: Content: ', '')
                content = content.strip()
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = ' '.join(content.split()[:int(max_length)])
                messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
                messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
            messages.append({'role': 'user', 'content': self._get_post_prompt(query, num)})
            num_tokens = self.get_num_tokens(messages)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(1, (num_tokens - self.max_tokens() + self.num_output_tokens()) // (rank_end- rank_start))    
        return messages, self.get_num_tokens(messages)

    def get_num_tokens(self, messages):
        """Returns the number of tokens used by a list of messages."""
        if self.model_ in ["gpt-3.5-turbo-0301",  "gpt-3.5-turbo"]:
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
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
    
    def cost_per_1k_token(self, input_token:bool):
        # Brought in from https://openai.com/pricing on 2023-07-30
        cost_dict = {
                ("gpt-3.5", 4096): 0.0015 if input_token else 0.002,
                ("gpt-3.5", 16384): 0.003 if input_token else 0.004,
                ("gpt-4", 8192): 0.03 if input_token else 0.06,
                ("gpt-4", 32768): 0.06 if input_token else 0.12
            }
        model_key = "gpt-3.5" if "gpt-3" in self.model_ else "gpt-4"
        return cost_dict[(model_key, self.context_size_)]

def _get_api_key():
    from dotenv import dotenv_values, load_dotenv
    import os
    load_dotenv(dotenv_path=f'.env.local')
    return os.getenv("OPEN_AI_API_KEY")

def main():
    openai_keys = _get_api_key()
    model_name='gpt-3.5-turbo'
    context_size = 4096
    dataset = 'dl20'
    prompt_mode = PromptMode.RANK_GPT
    agent = SafeOpenai(model=model_name, context_size=context_size, dataset=dataset, prompt_mode=prompt_mode, keys=openai_keys)

    retriever = PyseriniRetriever(dataset)
    from pathlib import Path

    candidates_file = Path(f'retrieve_results/retrieve_results_{dataset}.json')
    if not candidates_file.is_file():
        print('Retrieving:')
        retriever.retrieve_and_store(k=100)
    else:
        print('Reusing existing retrieved results.')
    import json
    with open(candidates_file, 'r') as f:
        retrieved_results = json.load(f)

    print('\nReranking:')
    rerank_results = []
    input_token_counts = []
    output_token_counts = []
    aggregated_prompts = []
    aggregated_responses = [] 
    for result in tqdm(retrieved_results):
        rerank_result, in_token_count, out_token_count, prompts, responses  = agent.sliding_windows(result, rank_start=0, rank_end=100, window_size=20, step=10)
        rerank_results.append(rerank_result)
        input_token_counts.append(in_token_count)
        output_token_counts.append(out_token_count)
        aggregated_prompts.extend(prompts)
        aggregated_responses.extend(responses)
    print(f'input_tokens_counts={input_token_counts}')
    print(f'total input token count={sum(input_token_counts)}')
    print(f'output_token_counts={output_token_counts}')
    print(f'total output token count={sum(output_token_counts)}')
    file_name = agent.write_rerank_results(rerank_results, input_token_counts, output_token_counts, aggregated_prompts, aggregated_responses)
    from trec_eval import EvalFunction
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', TOPICS[dataset], file_name])


if __name__ == '__main__':
    main()

import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .listwise_rankllm import ListwiseRankLLM

from tqdm import tqdm
import tiktoken
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from rank_llm.rerank import PromptMode
from rank_llm.data import Request, Result

'''
make sure to run
pip install -U -q "google-generativeai>=0.8.2"
'''


## required functions 

## 


class SafeGemini(ListwiseRankLLM):
    
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode = PromptMode.GEMINI,
        num_few_shot_examples: int = 0,
        window_size: int = 20,
        keys=None,
        key_start_id=None,
        temperature = 0,
        top_p = 0.95,
        top_k = 40,
        max_output_tokens = 8192
    ):
        super().__init__(
            model, context_size, prompt_mode, num_few_shot_examples, window_size
        )
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("Please provide Gemini Keys.")
        if prompt_mode not in [
            PromptMode.GEMINI,
        ]:
            raise ValueError(
                f"unsupported prompt mode for GPT models: {prompt_mode}, expected {PromptMode.GEMINI}."
            )
        self._output_token_estimate = None
        self._keys = keys
        self._cur_key_id = key_start_id or 0
        self._cur_key_id = self._cur_key_id % len(self._keys)
        self.generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "text/plain",
        }
        
        self.model_name = model
        self.safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}, # ask how to block civic integrity and add here
        ]
        self.system_instruction = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
        genai.configure(api_key=self._keys[self._cur_key_id])
    
    class CompletionMode(Enum):
        CHAT = 1
        
    
    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        window_size: int = kwargs.get("window_size", 20)
        step: int = kwargs.get("step", 10)
        populate_exec_summary: bool = kwargs.get("populate_exec_summary", False)

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

    def run_llm_batched(self):
        pass
    
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
                    model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=self.generation_config,
                    system_instruction=self.system_instruction,
                    safety_settings=self.safety_settings
                    )
                    chat_session = model.start_chat(
                    history=[
                    ]
                    )
                    completion = chat_session.send_message(kwargs.get("messages"))
                else:
                    raise ValueError(
                        "Unsupported completion mode: %V" % completion_mode
                    )
                break
            except Exception as e:
                print("Error in completion call")
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print("reduce_length")
                    return "ERROR::reduce_length"
                if "The response was filtered" in str(e):
                    print("The response was filtered")
                    return "ERROR::The response was filtered"
                self._cur_key_id = (self._cur_key_id + 1) % len(self._keys)
                self.api_key = self._keys[self._cur_key_id] 
                genai.configure(api_key=self.api_key)
                time.sleep(5.0)
        if return_text:
            completion = completion.text
        return completion

    def run_llm(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        current_window_size: Optional[int] = None,
    ) -> Tuple[str, int]:
        model_key = "model"
        response = self._call_completion(
            messages=prompt,
            temperature=1,
            completion_mode=SafeGemini.CompletionMode.CHAT,
            return_text=True,
            **{model_key: self._model},
        )
        token_counter = genai.GenerativeModel(self.model_name)
        return response, token_counter.count_tokens(response).total_tokens


    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[List[Dict[str, str]], int]:
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])
        max_length = 300 * (20 / (rank_end - rank_start))
        psg_ids = []
        while True:
            message = f"Given the query: [querystart] {query} [queryend], produce a succinct and clear ranking of all passages, from most to least relevant, using their identifiers. The format should be [rankstart] [most relevant passage NUM] > [next most relevant passage NUM] > ... > [least relevant passage NUM] [rankend]. ex: [rankstart] [1] > [2] ... [{num}] [rankend] Refrain from including any additional commentary or explanations in your ranking."
            rank = 0
            for cand in result.candidates[rank_start:rank_end]:
                rank += 1
                psg_id = f"PASSAGE{rank}"
                content = self.convert_doc_to_prompt_content(cand.doc, max_length)
                message += f'{psg_id} = "{self._replace_number(content)}"\n'
                psg_ids.append(psg_id)
            message += f'QUESTION = "{query}"\n'
            message += "PASSAGES = [" + ", ".join(psg_ids) + "]\n"
            message += "SORTED_PASSAGES = [\n"
            message = {
                        "parts": [
                            {
                                "text": message
                            }
                        ]
                        }
            #message = [{"role": "user", "content": message}]
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
            token_counter = genai.GenerativeModel(self.model_name)

            _output_token_estimate = token_counter.count_tokens("[rankstart] " + " > ".join([f"[{i+1}]" for i in range(current_window_size)]) + " [rankend]").total_tokens - 1
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
        model = genai.GenerativeModel(self.model_name)
        if isinstance(prompt, list):
            for message in prompt:
                for key, value in message.items():
                    response = model.count_tokens(value).total_tokens
                    num_tokens += response
        else:
                response = model.count_tokens(prompt).total_tokens
                num_tokens += response
        # num_tokens += 3  # every reply is primed with <|start|>assistant<|message|> check later for how to approch this issue
        return num_tokens
    
    
    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
    
    def get_name(self) -> str:
        return self.model_name
    
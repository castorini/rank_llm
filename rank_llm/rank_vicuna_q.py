import re
from typing import Tuple, List, Union, Dict, Any

from fastchat.model import load_model, get_conversation_template, add_model_args
from ftfy import fix_text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

from rank_llm import RankLLM, PromptMode
from llama_cpp import Llama, LlamaCache


def replace_number(s):
    return re.sub(r"\[(\d+)\]", r"(\1)", s)


class RankVicunaQ(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int,
        top_k_candidates: int,
        dataset: str,
        prompt_mode: PromptMode,
        device: str,
        num_gpus: int = 1,  # AFAIK, support for multiple GPUS is not very good at Llama.cpp.
    ) -> None:
        super().__init__(model, context_size, top_k_candidates, dataset, prompt_mode)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsuported prompt mode: {prompt_mode}. The only prompt mode cuurently supported by vicuna is a slight variation of Rank_GPT prompt."
            )
        self._llm = Llama(
            model_path=model,
            n_ctx=context_size,
            n_gpu_layers=-1,
            verbose=False,
        )
        self._llm.set_cache(LlamaCache())

    def run_llm(self, prompt: str) -> Tuple[str, int]:
        output: Dict[str, Any] = self._llm(
            prompt, max_tokens=self.max_tokens(), temperature=0.9, top_p=0.6
        )  # type: ignore
        text = output["choices"][0]["text"]
        n_tokens = output["usage"]["completion_tokens"]
        return text, n_tokens

    def num_output_tokens(self) -> int:
        return 200

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, query: str, num: int) -> str:
        return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."

    def create_prompt(
        self, retrieved_result: Dict[str, Any], rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        query = retrieved_result["query"]
        num = len(retrieved_result["hits"][rank_start:rank_end])
        max_length = 300
        while True:
            conv = get_conversation_template(self._model)
            # conv.set_system_message(
            #     "You are RankVicuna, an intelligent assistant that can rank passages based on their relevancy to the query."
            # )
            prefix = self._add_prefix_prompt(query, num)
            rank = 0
            input_context = f"{prefix}\n"
            for hit in retrieved_result["hits"][rank_start:rank_end]:
                rank += 1
                content = hit["content"]
                content = content.replace("Title: Content: ", "")
                content = content.strip()
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = " ".join(content.split()[: int(max_length)])
                input_context += f"[{rank}] {replace_number(content)}\n"

            input_context += self._add_post_prompt(query, num)
            conv.append_message(conv.roles[0], input_context)
            prompt = conv.get_prompt() + " ASSISTANT:"
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // (rank_end - rank_start),
                )
        return prompt, self.get_num_tokens(prompt)

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._llm.tokenize(prompt.encode()))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

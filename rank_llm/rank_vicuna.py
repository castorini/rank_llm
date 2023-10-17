import re
from typing import Tuple, List, Union, Dict, Any

from fastchat.model import load_model, get_conversation_template, add_model_args
from ftfy import fix_text
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

from rank_llm.rankllm import RankLLM, PromptMode
from rank_llm.topics_dict import TOPICS

def replace_number(s):
    return re.sub(r"\[(\d+)\]", r"(\1)", s)


class RankVicuna(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        device: str = "cuda",
        num_gpus: int = 1,
    ) -> None:
        super().__init__(model, context_size, prompt_mode)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsuported prompt mode: {prompt_mode}. The only prompt mode cuurently supported by vicuna is a slight variation of Rank_GPT prompt."
            )
        # ToDo: Make repetition_penalty configurable
        self._llm, self._tokenizer = load_model(model, device=device, num_gpus=num_gpus)

    def run_llm(self, prompt: str) -> Tuple[str, int]:
        inputs = self._tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items()}
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens()
        gen_cfg.min_length = 1
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
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

    def rerank(
        self,
        query: str,
        documents: Union[List[str], List[Dict[str, Any]]],
    ):
        assert len(documents) > 0, "'documents' should be non-empty"

        print("Reranking with RankVicuna:")
        rerank_results = []
        input_token_counts = []
        output_token_counts = []
        aggregated_prompts = []
        aggregated_responses = []

        if isinstance(documents[0], str):
            document_hits = []
            for passage in documents:
                document_hits.append({
                    "content": passage
                })
            retrieved_result = [{
                "query": query,
                "hits": document_hits,
            }]
        elif isinstance(documents[0], dict):
            retrieved_result = [{
                "query": query,
                "hits": documents,
            }]

        for result in tqdm(retrieved_result):
            (
                rerank_result,
                in_token_count,
                out_token_count,
                prompts,
                responses,
            ) = self.sliding_windows(
                result,
                rank_start=0,
                rank_end=len(documents),
                window_size=min(len(documents), 20),
                step=10,
                shuffle_candidates=False,
                logging=True,
            )
            rerank_results.append(rerank_result)
            input_token_counts.append(in_token_count)
            output_token_counts.append(out_token_count)
            aggregated_prompts.extend(prompts)
            aggregated_responses.extend(responses)
        
        print(f"rerank_results={rerank_result}")
        print(f"input_tokens_counts={input_token_counts}")
        print(f"total input token count={sum(input_token_counts)}")
        print(f"output_token_counts={output_token_counts}")
        print(f"total output token count={sum(output_token_counts)}")

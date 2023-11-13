import json
import random
import re
from typing import Tuple, Dict, Any

from fastchat.model import get_conversation_template
from ftfy import fix_text

from rank_llm.rankllm import RankLLM, PromptMode
from rank_llm.rank_vicuna import RankVicuna

def replace_number(s):
    return re.sub(r"\[(\d+)\]", r"(\1)", s)


class RankZephyr(RankVicuna):
    def create_prompt(
        self, retrieved_result: Dict[str, Any], rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        query = retrieved_result["query"]
        num = len(retrieved_result["hits"][rank_start:rank_end])
        max_length = 300
        while True:
            conv = get_conversation_template(self._model)
            conv.set_system_message("You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.")
            conv = self._add_few_shot_examples(conv)
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
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // ((rank_end - rank_start) * 4),
                )
        return prompt, self.get_num_tokens(prompt)

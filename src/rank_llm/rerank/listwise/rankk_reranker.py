from typing import Tuple
import os

from ftfy import fix_text

from rank_llm.data import Request, Result
from rank_llm.rerank import PromptMode
from rank_llm.rerank.listwise import RankListwiseOSLLM

ALPH_START_IDX = ord("A") - 1

rank_k_prompt = """
Determine a ranking of the passages based on how relevant they are to the query. 
If the query is a question, how relevant a passage is depends on how well it answers the question. 
If not, try analyze the intent of the query and assess how well each passage satisfy the intent. 
The query may have typos and passages may contain contradicting information. 
However, we do not get into fact-checking. We just rank the passages based on they relevancy to the query. 

Sort them from the most relevant to the least. 
Answer with the passage number using a format of `[3] > [2] > [4] = [1] > [5]`. 
Ties are acceptable if they are equally relevant. 
I need you to be accurate but overthinking it is unnecessary.
Output only the ordering without any other text.

Query: {query}

{docs}
"""

class RankKReranker(RankListwiseOSLLM):
    def __init__(
            self,
            model: str = "hltcoe/Rank-K-32B",
            context_size: int = 4096,
            prompt_mode: PromptMode = PromptMode.RANK_GPT,
            num_few_shot_examples: int = 0,
            device: str = "cuda",
            num_gpus: int = 1,
            variable_passages: bool = True,
            window_size: int = 20,
            use_alpha: bool = False
    ) -> None:
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples = num_few_shot_examples,
            device = device,
            num_gpus=num_gpus,
            variable_passages= variable_passages,
            window_size=window_size,
            use_alpha=use_alpha)
    
    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        query = result.query.text

        rank = 0
        input_context=""
        for cand in result.candidates[rank_start:rank_end]:
            rank += 1
            content = cand.doc[list(cand.doc.keys())[0]]

            identifier = str(rank)
            input_context += f"[{identifier}] {self._replace_number(content)}\n\n"

        input_context = input_context.strip()
        content = rank_k_prompt.format(
            query=query,
            docs=input_context
        )
        messages=[{"role": "user", "content": content}]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
            )
        
        return prompt, self.get_num_tokens(prompt)


    
    #Score: 0.6944, 0.5327, 0.5014 still  bugged
    # def create_prompt(
    #     self, result: Result, rank_start: int, rank_end: int
    # ) -> Tuple[str, int]:
    #     query = result.query.text
    #     query = self._replace_number(query)

    #     max_length = 300 * (20 / (rank_end - rank_start))
    #     input_context=""
    #     while True:
    #         rank = 0
    #         for cand in result.candidates[rank_start:rank_end]:
    #             rank += 1
    #             content = self.convert_doc_to_prompt_content(cand.doc, max_length)

    #             identifier = (
    #                 chr(ALPH_START_IDX + rank) if self._use_alpha else str(rank)
    #             )
    #             input_context += f"[{identifier}] {self._replace_number(content)}\n\n"
    #         input_context = input_context.strip()
    #         content = rank_k_prompt.format(
    #             query=query,
    #             docs=input_context
    #         )
    #         messages=[{"role": "user", "content": content}]
    #         prompt = self._tokenizer.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=False
    #             )
        
    #         #prompt = fix_text(prompt)
    #         num_tokens = self.get_num_tokens(prompt)
    #         if num_tokens <= self.max_tokens() - self.num_output_tokens(
    #             rank_end - rank_start
    #         ):
    #             break
    #         else:
    #             max_length -= max(
    #                 1,
    #                 (
    #                     num_tokens
    #                     - self.max_tokens()
    #                     + self.num_output_tokens(rank_end - rank_start, self._use_alpha)
    #                 )
    #                 // ((rank_end - rank_start) * 4),
    #             )
    #     return prompt, self.get_num_tokens(prompt)
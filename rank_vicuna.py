from rank_llm import RankLLM, PromptMode
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template, add_model_args
from tqdm import tqdm
from pyserini_retriever import PyseriniRetriever, RetrievalMethod
from topics_dict import TOPICS
from transformers.generation import GenerationConfig


class RankVicuna(RankLLM):
    def __init__(self, model, context_size, dataset, prompt_mode, device):
        super().__init__(model, context_size, dataset, prompt_mode)
        self.device_ = device
        if self.device_ == "cuda":
            assert torch.cuda.is_available()
        # ToDo: Make repetition_penalty configurable
        self.llm_, self.tokenizer_ = load_model(model, device=device)

    def run_llm(self, messages):
        inputs = self.tokenizer_([messages])
        inputs = {k: torch.tensor(v).to(self.device_) for k, v in inputs.items()}
        self.llm_.config.max_new_tokens = self.num_output_tokens()
        self.llm_.config.min_length = 1
        self.llm_.config.temperature = 0
        self.llm_.config.do_sample = False
        gen_cfg = GenerationConfig.from_model_config(self.llm_.config)
        gen_cfg.max_new_tokens = 10
        gen_cfg.min_length = 1
        gen_cfg.temperature = 0
        gen_cfg.do_sample = False
        output_ids = self.llm_.generate(**inputs, generation_config=gen_cfg)

        if self.llm_.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self.tokenizer_.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        print(f"outputs: {outputs}")
        return outputs, len(self.tokenizer_.encode(outputs))

    def num_output_tokens(self):
        return 200

    def _add_prefix_prompt(self, query, num, conv):
        conv.set_system_message(
            "You are Vicuna, an intelligent assistant that can rank passages based on their relevancy to the query."
        )
        conv.append_message(
            conv.roles[0],
            f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
        )
        conv.append_message(conv.roles[1], "Okay, please provide the passages.")

    def _add_post_prompt(self, query, num, conv):
        conv.append_message(conv.roles[1], f"Okay, {num} passages are provided.")
        conv.append_message(
            conv.roles[0],
            f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.",
        )

    def create_prompt(self, retrieved_result, rank_start=0, rank_end=100):
        query = retrieved_result["query"]
        num = len(retrieved_result["hits"][rank_start:rank_end])

        max_length = 300
        while True:
            conv = get_conversation_template(self.model_)
            self._add_prefix_prompt(query, num, conv)
            rank = 0
            for hit in retrieved_result["hits"][rank_start:rank_end]:
                rank += 1
                content = hit["content"]
                content = content.replace("Title: Content: ", "")
                content = content.strip()
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = " ".join(content.split()[: int(max_length)])
                conv.append_message(conv.roles[0], f"[{rank}] {content}")
                conv.append_message(conv.roles[1], f"Received passage [{rank}].")
            self._add_post_prompt(query, num, conv)
            prompt = conv.get_prompt()
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

    def get_num_tokens(self, messages):
        return len(self.tokenizer_.encode(messages))

    def cost_per_1k_token(self, input_token: bool):
        return 0


def main():
    # model_path='lmsys/vicuna-13b-v1.5'

    context_size = 4096
    dataset = "dl19"
    prompt_mode = PromptMode.RANK_GPT
    device = "cuda"
    agent = RankVicuna(model_path, context_size, dataset, prompt_mode, device)
    retrieval_method = RetrievalMethod.BM25
    retriever = PyseriniRetriever(dataset, retrieval_method)
    from pathlib import Path

    candidates_file = Path(
        f"retrieve_results/{retrieval_method.name}/retrieve_results_{dataset}.json"
    )
    if not candidates_file.is_file():
        print("Retrieving:")
        retriever.retrieve_and_store(k=100)
    else:
        print("Reusing existing retrieved results.")
    import json

    with open(candidates_file, "r") as f:
        retrieved_results = json.load(f)

    print("\nReranking:")
    rerank_results = []
    input_token_counts = []
    output_token_counts = []
    aggregated_prompts = []
    aggregated_responses = []
    for result in tqdm(retrieved_results):
        (
            rerank_result,
            in_token_count,
            out_token_count,
            prompts,
            responses,
        ) = agent.sliding_windows(
            result, rank_start=0, rank_end=100, window_size=20, step=10
        )
        rerank_results.append(rerank_result)
        input_token_counts.append(in_token_count)
        output_token_counts.append(out_token_count)
        aggregated_prompts.extend(prompts)
        aggregated_responses.extend(responses)
    print(f"input_tokens_counts={input_token_counts}")
    print(f"total input token count={sum(input_token_counts)}")
    print(f"output_token_counts={output_token_counts}")
    print(f"total output token count={sum(output_token_counts)}")
    file_name = agent.write_rerank_results(
        retrieval_method.name,
        rerank_results,
        input_token_counts,
        output_token_counts,
        aggregated_prompts,
        aggregated_responses,
    )
    from trec_eval import EvalFunction

    EvalFunction.eval(["-c", "-m", "ndcg_cut.10", TOPICS[dataset], file_name])


if __name__ == "__main__":
    main()

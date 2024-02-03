import json
import os
import sys
from argparse import ArgumentParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.analysis.estimate_costs import EstimationMode
from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.retrieve.pyserini_retriever import PyseriniRetriever, RetrievalMethod
from rank_llm.retrieve.topics_dict import TOPICS


def main(args):
    estimation_mode = args.estimation_mode
    openai_keys = "Fake key, not needed for cost estimate"
    model_name = args.model_name
    context_size = args.context_size
    top_k_candidates = args.top_k_candidates
    num_few_shot_examples = args.num_few_shot_examples
    retrieval_method = RetrievalMethod.BM25
    prompt_mode = args.prompt_mode
    costs = {}
    for dataset in TOPICS.keys():
        print("#" * 20)
        retriever = PyseriniRetriever(dataset, retrieval_method)
        num_queries = retriever.num_queries()
        agent = SafeOpenai(
            model=model_name,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            keys=openai_keys,
        )
        print(
            f'Estimating cost for "{dataset}" with "{retrieval_method.value}" retrieval method, {top_k_candidates} top candidates, {context_size} context_size, and "{model_name}" model with {agent.cost_per_1k_token(input_token=True)}|{agent.cost_per_1k_token(input_token=False)} per input|output 1k tokens:'
        )
        if estimation_mode == EstimationMode.CREATE_PROMPTS:
            print("Reterieving candidates:")
            retrieved_results = retriever.retrieve(k=top_k_candidates)
            # For dl20 the number of retrieved results is different from the number of queries/topics.
            num_queries = len(retrieved_results)
            print("Estimating cost by prompt generation:")
            cost, token_count = agent.get_ranking_cost(
                retrieved_results, rank_start=0, rank_end=100, window_size=20, step=10
            )
        elif estimation_mode == EstimationMode.MAX_CONTEXT_LENGTH:
            cost, token_count = agent.get_ranking_cost_upperbound(
                num_queries, rank_start=0, rank_end=100, window_size=20, step=10
            )
        else:
            raise ValueError(f"Invalide estimation mode: {estimation_mode}")
        costs[str((dataset, num_queries, token_count))] = cost
        print(f"The cost is {cost} USD for {token_count} tokens.")
        print("#" * 20)
        print("\n\n")
    print(
        f"The cost dict for {model_name} with {context_size} context size is {json.dumps(costs, sort_keys=True, indent=4)}.\n"
    )
    total_cost = sum(cost for cost in costs.values())
    print(f"The total estimated cost is {total_cost}.\n")


"""
python src/rank_llm/analysis/estimate_costs.py --estimation_mode=create_prpts --model_name=gpt-3.5-turbo --prompt_mode=rank_GPT
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--estimation_mode",
        type=EstimationMode,
        choices=list(EstimationMode),
        required=True,
        help="""The estimation mode: 
                                    `max_context_length` for simply using the max context length for each prompt or
                                    `create_prompts` for calculating cost estimates by using a sliding window to create prompts""",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="name of the model used for price estimation",
    )
    parser.add_argument(
        "--context_size", type=int, default=4096, help="context size used for model"
    )
    parser.add_argument(
        "--top_k_candidates",
        type=int,
        default=100,
        help="the number of top candidates to rerank",
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=0,
        help="the number of examples provided in prompt",
    )
    parser.add_argument(
        "--prompt_mode",
        type=PromptMode,
        required=True,
        choices=list(PromptMode),
    )
    args = parser.parse_args()
    main(args)

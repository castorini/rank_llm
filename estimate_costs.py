import json
from argparse import ArgumentParser
from enum import Enum
from pyserini_retriever import PyseriniRetriever
from rank_gpt import SafeOpenai
from rank_llm import PromptMode
from topics_dict import TOPICS


class EstimationMode(Enum):
    MAX_CONTEXT_LENGTH = 'max_context_length'
    CREATE_PROMPTS = 'create_prompts'

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EstimationMode[s]
        except KeyError:
            raise ValueError()


def main():
    parser = ArgumentParser()
    parser.add_argument('--estimation_mode', type=EstimationMode.from_string, choices=list(EstimationMode), required=True, 
                        help="""The estimation mode: 
                                    `MAX_CONTEXT_LENGTH` for simply using the max context length for each prompt or
                                    `CREATE_PROMPTS` for calculating cost estimates by using a sliding window to create prompts""")
    estimation_mode = parser.parse_args().estimation_mode
    costs = {}
    openai_keys = "Fake key"  # Your openai key
    model_name='gpt-3.5-turbo'
    context_size = 4096
    prompt_mode = PromptMode.RANK_GPT
    for dataset in TOPICS.keys():
        print('#' * 20)
        retriever = PyseriniRetriever(dataset)
        num_queries = retriever.num_queries()
        agent = SafeOpenai(model=model_name, context_size=context_size, dataset=dataset,prompt_mode=prompt_mode, keys=openai_keys)
        print(f'Estimating cost for "{dataset}" with {context_size} context_size and "{model_name}" model with {agent.cost_per_1k_token(input_token=True)}|{agent.cost_per_1k_token(input_token=False)} per input|output 1k tokens:')
        if estimation_mode == EstimationMode.CREATE_PROMPTS:
            print('Reterieving candidates:')
            retrieved_results = retriever.retrieve(k=100)
            # For dl20 the number of retrieved results is different from the number of queries/topics.
            num_queries = len(retrieved_results)
            print('Estimating cost by prompt generation:')
            cost, token_count = agent.get_ranking_cost(retrieved_results, rank_start=0, rank_end=100, window_size=20, step=10)
        elif estimation_mode == EstimationMode.MAX_CONTEXT_LENGTH:
            cost, token_count = agent.get_ranking_cost_upperbound(num_queries, rank_start=0, rank_end=100, window_size=20, step=10)
        else:
            raise ValueError(f'Invalide estimation mode: {estimation_mode}')
        costs[str((dataset, num_queries, token_count))] = cost
        print(f'The cost is {cost} USD for {token_count} tokens.')
        print('#' * 20)
        print('\n\n')
    print(f'The cost dict for {model_name} with {context_size} context size is {json.dumps(costs, sort_keys=True, indent=4)}.\n')
    total_cost = sum(cost for cost in costs.values())
    print(f'The total estimated cost is {total_cost}.\n')


if __name__ == '__main__':
    main()
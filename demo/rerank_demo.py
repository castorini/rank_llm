from rank_llm.rank_vicuna import RankVicuna
from rank_llm.rankllm import PromptMode

rv = RankVicuna(model="castorini/rank_vicuna_7b_v1", context_size=2000, top_k_candidates=4, dataset="dl19", prompt_mode=PromptMode.RANK_GPT, device="cuda", num_gpus=1)

# query = 'how long is life cycle of flea'
# docs = [
#       {
#         "content": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day. The flea egg stage is the beginning of the flea cycle. This part of the flea cycle represents a little more than one third of the flea population at any given time. Depending on the temperature and humidity of the environment the egg can take from two to six days to hatch.",
#         "qid": 264014,
#         "docid": "6641238",
#         "rank": 1,
#         "score": 3.605299949645996
#       },
#       {
#         "content": "2) The fleas life cycle discussed - the flea life cycle diagram explained in full. 2a) Fleas life cycle 1 - The adult flea lays her eggs on the host animal. 2b) Fleas life cycle 2 - The egg falls off the animal's skin and into the local environment of the host animal. 2c) Fleas life cycle 3 - The flea egg hatches, releasing a first stage (stage 1) flea larva.",
#         "qid": 264014,
#         "docid": "96854",
#         "rank": 2,
#         "score": 3.5165998935699463
#       },
#       {
#         "content": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day.",
#         "qid": 264014,
#         "docid": "4834547",
#         "rank": 3,
#         "score": 3.46589994430542
#       },
#       {
#         "content": "In appearance, flea larvae can be up to \u00c2\u00bc-inch long and are white (almost see-through) and legless. Larvae make up about 35 percent of the flea population in the average household. If conditions are favorable, the larvae will spin cocoons in about 5-20 days of hatching from their eggs.This leads to the next life stage, called the cocoon or pupae stage.The pupae stage of the flea life cycle accounts for about 10 percent of the flea population in a home.f conditions are favorable, the larvae will spin cocoons in about 5-20 days of hatching from their eggs. This leads to the next life stage, called the cocoon or pupae stage. The pupae stage of the flea life cycle accounts for about 10 percent of the flea population in a home.",
#         "qid": 264014,
#         "docid": "5635521",
#         "rank": 4,
#         "score": 3.4605000019073486
#       }]

query = 'What is the capital of the United States?'
docs = ['Carson City is the capital city of the American state of Nevada.',
     'The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.',
     'Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.',
     'Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.'
     ]

results = rv.rerank(query=query, documents=docs)

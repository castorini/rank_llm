from rank_llm.rank_vicuna import RankVicuna

rv = RankVicuna(model="castorini/rank_vicuna_7b_v1")
query = 'What is the capital of the United States?'
docs = ['Carson City is the capital city of the American state of Nevada.',
     'The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.',
     'Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.',
     'Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.'
     ]
results = rv.rerank(query=query, documents=docs)

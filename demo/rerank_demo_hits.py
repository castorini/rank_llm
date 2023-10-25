from rank_llm.retrieve_and_rerank import retrieve_and_rerank
from rank_llm.retriever import RetrievalMode
from rank_llm.pyserini_retriever import RetrievalMethod

query = 'how long is life cycle of flea'
hits = [
      {
        "content": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day.",
        "qid": 264014,
        "docid": "4834547",
        "rank": 3,
        "score": 14.971799850463867
      },
      {
        "content": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day. The flea egg stage is the beginning of the flea cycle. This part of the flea cycle represents a little more than one third of the flea population at any given time. Depending on the temperature and humidity of the environment the egg can take from two to six days to hatch.",
        "qid": 264014,
        "docid": "6641238",
        "rank": 2,
        "score": 15.090800285339355
      },
      {
        "content": "To go to our more detailed flea life cycle and flea control page, click here. 1) The flea life cycle diagram - a complete step-by-step diagram of animal host flea infestation; flea reproduction and environmental flea contamination with juvenile flea life cycle stages (eggs, larvae and pupae).",
        "qid": 264014,
        "docid": "1610712",
        "rank": 8,
        "score": 13.455499649047852
      },
      {
        "content": "Flea Pupa. The flea larvae spin cocoons around themselves in which they move to the last phase of the flea life cycle and become adult fleas. The larvae can remain in the cocoon anywhere from one week to one year. Temperature is one factor that determines how long it will take for the adult flea to emerge from the cocoon.",
        "qid": 264014,
        "docid": "96852",
        "rank": 4,
        "score": 14.215100288391113
      },
      {
        "content": "The cat flea's primary host is the domestic cat, but it is also the primary flea infesting dogs in most of the world. The cat flea can also maintain its life cycle on other carnivores and on omnivores. Humans can be bitten, though a long-term population of cat fleas cannot be sustained and infest people.However, if the female flea is allowed to feed for 12 consecutive hours on a human, it can lay viable eggs.he cat flea's primary host is the domestic cat, but it is also the primary flea infesting dogs in most of the world. The cat flea can also maintain its life cycle on other carnivores and on omnivores. Humans can be bitten, though a long-term population of cat fleas cannot be sustained and infest people.",
        "qid": 264014,
        "docid": "4239616",
        "rank": 6,
        "score": 13.947500228881836
      },
      {
        "content": "5. Cancel. A flea can live up to a year, but its general lifespan depends on its living conditions, such as the availability of hosts. Find out how long a flea's life cycle can last with tips from a pet industry specialist in this free video on fleas and pest control.Part of the Video Series: Flea Control.ancel. A flea can live up to a year, but its general lifespan depends on its living conditions, such as the availability of hosts. Find out how long a flea's life cycle can last with tips from a pet industry specialist in this free video on fleas and pest control. Part of the Video Series: Flea Control.",
        "qid": 264014,
        "docid": "5611210",
        "rank": 1,
        "score": 15.780599594116211
      },
      {
        "content": "2) The fleas life cycle discussed - the flea life cycle diagram explained in full. 2a) Fleas life cycle 1 - The adult flea lays her eggs on the host animal. 2b) Fleas life cycle 2 - The egg falls off the animal's skin and into the local environment of the host animal. 2c) Fleas life cycle 3 - The flea egg hatches, releasing a first stage (stage 1) flea larva.",
        "qid": 264014,
        "docid": "96854",
        "rank": 5,
        "score": 13.985199928283691
      },
      {
        "content": "In appearance, flea larvae can be up to \u00c2\u00bc-inch long and are white (almost see-through) and legless. Larvae make up about 35 percent of the flea population in the average household. If conditions are favorable, the larvae will spin cocoons in about 5-20 days of hatching from their eggs.This leads to the next life stage, called the cocoon or pupae stage.The pupae stage of the flea life cycle accounts for about 10 percent of the flea population in a home.f conditions are favorable, the larvae will spin cocoons in about 5-20 days of hatching from their eggs. This leads to the next life stage, called the cocoon or pupae stage. The pupae stage of the flea life cycle accounts for about 10 percent of the flea population in a home.",
        "qid": 264014,
        "docid": "5635521",
        "rank": 7,
        "score": 13.533599853515625
      }]

results = retrieve_and_rerank(
    model_path="castorini/rank_vicuna_7b_v1",
    dataset=hits,    
    retrieval_mode=RetrievalMode.QUERY_AND_HITS,
    retrieval_method=RetrievalMethod.UNSPECIFIED,
    top_k_candidates=len(hits),
    print_prompts_responses=True,
    query=query
)

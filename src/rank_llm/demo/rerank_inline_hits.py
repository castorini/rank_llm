import os
import sys
from pathlib import Path

from dacite import from_dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, Request
from rank_llm.rerank.listwise import VicunaReranker, ZephyrReranker

request_dict = {
    "query": {"text": "how long is life cycle of flea", "qid": "264014"},
    "candidates": [
        {
            "doc": {
                "segment": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day."
            },
            "docid": "4834547",
            "score": 14.971799850463867,
        },
        {
            "doc": {
                "segment": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day. The flea egg stage is the beginning of the flea cycle. This part of the flea cycle represents a little more than one third of the flea population at any given time. Depending on the temperature and humidity of the environment the egg can take from two to six days to hatch."
            },
            "docid": "6641238",
            "score": 15.090800285339355,
        },
        {
            "doc": {
                "segment": "To go to our more detailed flea life cycle and flea control page, click here. 1) The flea life cycle diagram - a complete step-by-step diagram of animal host flea infestation; flea reproduction and environmental flea contamination with juvenile flea life cycle stages (eggs, larvae and pupae)."
            },
            "docid": "1610712",
            "score": 13.455499649047852,
        },
        {
            "doc": {
                "segment": "Flea Pupa. The flea larvae spin cocoons around themselves in which they move to the last phase of the flea life cycle and become adult fleas. The larvae can remain in the cocoon anywhere from one week to one year. Temperature is one factor that determines how long it will take for the adult flea to emerge from the cocoon."
            },
            "docid": "96852",
            "score": 14.215100288391113,
        },
        {
            "doc": {
                "segment": "The cat flea's primary host is the domestic cat, but it is also the primary flea infesting dogs in most of the world. The cat flea can also maintain its life cycle on other carnivores and on omnivores. Humans can be bitten, though a long-term population of cat fleas cannot be sustained and infest people.However, if the female flea is allowed to feed for 12 consecutive hours on a human, it can lay viable eggs.he cat flea's primary host is the domestic cat, but it is also the primary flea infesting dogs in most of the world. The cat flea can also maintain its life cycle on other carnivores and on omnivores. Humans can be bitten, though a long-term population of cat fleas cannot be sustained and infest people."
            },
            "docid": "4239616",
            "score": 13.947500228881836,
        },
        {
            "doc": {
                "segment": "5. Cancel. A flea can live up to a year, but its general lifespan depends on its living conditions, such as the availability of hosts. Find out how long a flea's life cycle can last with tips from a pet industry specialist in this free video on fleas and pest control.Part of the Video Series: Flea Control.ancel. A flea can live up to a year, but its general lifespan depends on its living conditions, such as the availability of hosts. Find out how long a flea's life cycle can last with tips from a pet industry specialist in this free video on fleas and pest control. Part of the Video Series: Flea Control."
            },
            "docid": "5611210",
            "score": 15.780599594116211,
        },
        {
            "doc": {
                "segment": "2) The fleas life cycle discussed - the flea life cycle diagram explained in full. 2a) Fleas life cycle 1 - The adult flea lays her eggs on the host animal. 2b) Fleas life cycle 2 - The egg falls off the animal's skin and into the local environment of the host animal. 2c) Fleas life cycle 3 - The flea egg hatches, releasing a first stage (stage 1) flea larva."
            },
            "docid": "96854",
            "score": 13.985199928283691,
        },
        {
            "doc": {
                "segment": "In appearance, flea larvae can be up to \u00c2\u00bc-inch long and are white (almost see-through) and legless. Larvae make up about 35 percent of the flea population in the average household. If conditions are favorable, the larvae will spin cocoons in about 5-20 days of hatching from their eggs.This leads to the next life stage, called the cocoon or pupae stage.The pupae stage of the flea life cycle accounts for about 10 percent of the flea population in a home.f conditions are favorable, the larvae will spin cocoons in about 5-20 days of hatching from their eggs. This leads to the next life stage, called the cocoon or pupae stage. The pupae stage of the flea life cycle accounts for about 10 percent of the flea population in a home."
            },
            "docid": "5635521",
            "score": 13.533599853515625,
        },
    ],
}

request = from_dict(data_class=Request, data=request_dict)
reranker = ZephyrReranker()
rerank_results = reranker.rerank(request=request)
reranker = VicunaReranker()
rerank_results = reranker.rerank(request=request)
print(rerank_results)

# write rerank results
writer = DataWriter(rerank_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_json_format(f"demo_outputs/rerank_results.json")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_ranking_exec_summary(f"demo_outputs/ranking_execution_summary.json")

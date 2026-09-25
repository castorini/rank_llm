import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dacite import from_dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.data import DataWriter, Request
from rank_llm.rerank.listwise import VicunaReranker, ZephyrReranker

DEFAULT_MODELS = {
    "zephyr": "castorini/rank_zephyr_7b_v1_full",
    "vicuna": "castorini/rank_vicuna_7b_v1",
}

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerank a fixed set of inline passages with RankZephyr or RankVicuna."
    )
    parser.add_argument(
        "--rerankers",
        choices=tuple(DEFAULT_MODELS),
        nargs="+",
        default=("zephyr", "vicuna"),
        help="Rerankers to run in order (default: zephyr vicuna).",
    )
    parser.add_argument(
        "--zephyr-model",
        default=DEFAULT_MODELS["zephyr"],
        help=f"RankZephyr model ID (default: {DEFAULT_MODELS['zephyr']}).",
    )
    parser.add_argument(
        "--vicuna-model",
        default=DEFAULT_MODELS["vicuna"],
        help=f"RankVicuna model ID (default: {DEFAULT_MODELS['vicuna']}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=len(request_dict["candidates"]),
        help="Number of inline candidates to rerank (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        default="demo_outputs/inline_hits",
        help="Base output directory; each reranker writes to its own model directory.",
    )
    parser.add_argument(
        "--populate-invocations-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _make_reranker(name: str, args: argparse.Namespace):
    model_path = args.zephyr_model if name == "zephyr" else args.vicuna_model
    reranker_class = ZephyrReranker if name == "zephyr" else VicunaReranker
    return model_path, reranker_class(
        model_path=model_path,
        context_size=args.context_size,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.k <= len(request_dict["candidates"]):
        parser.error(f"--k must be between 1 and {len(request_dict['candidates'])}")
    if not 0 < args.stride <= args.window_size:
        parser.error("--stride must be greater than 0 and no larger than --window-size")

    for name in args.rerankers:
        request = from_dict(data_class=Request, data=request_dict)
        request.candidates = request.candidates[: args.k]
        model_path, reranker = _make_reranker(name, args)
        try:
            rerank_results = reranker.rerank(
                request=request,
                rank_end=args.k,
                top_k_retrieve=args.k,
                populate_invocations_history=args.populate_invocations_history,
            )
        finally:
            # vLLM runs the engine in a subprocess; close it before another model.
            reranker.close()

        print(f"{name} results: {rerank_results}")
        out_path = Path(args.output_dir) / model_path.split("/")[-1].lower()
        out_path.mkdir(parents=True, exist_ok=True)
        writer = DataWriter(rerank_results)
        writer.write_in_jsonl_format(str(out_path / "rerank.jsonl"))
        writer.write_in_trec_eval_format(str(out_path / "rerank.txt"))
        if args.populate_invocations_history:
            writer.write_inference_invocations_history(
                str(out_path / "invocations.json")
            )


if __name__ == "__main__":
    main()

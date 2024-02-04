import json
import os
from argparse import ArgumentParser

from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod
from rank_llm.retrieve.topics_dict import TOPICS


def main(args):
    # TODO: make metrics configurable
    model = args.model_name
    context_size = args.context_size
    prompt_mode = args.prompt_mode
    rerank_results_dirname = args.rerank_results_dirname
    output_filename = f"trec_eval_aggregated_results_{model}_{prompt_mode}.jsonl"
    with open(output_filename, "w") as output:
        for dataset in ["dl19", "dl20", "dl21", "dl22", "news", "covid"]:
            for retrieval_method in RetrievalMethod:
                if retrieval_method == RetrievalMethod.UNSPECIFIED:
                    continue
                directory = f"{rerank_results_dirname}/{retrieval_method.name}"
                if not os.path.isdir(directory):
                    continue
                for top_k_canidadates in [20, 100]:
                    for filename in os.listdir(directory):
                        if not filename.startswith(
                            f"{model}_{context_size}_{top_k_canidadates}_{prompt_mode}_{dataset}"
                        ):
                            continue
                        if filename.endswith(".json"):
                            continue
                        f = os.path.join(directory, filename)
                        # checking if it is a file
                        if os.path.isfile(f):
                            json.dump(
                                {
                                    "file": f,
                                    "result": [
                                        EvalFunction.eval(
                                            [
                                                "-c",
                                                "-m",
                                                "ndcg_cut.10",
                                                TOPICS[dataset],
                                                f,
                                            ]
                                        ),
                                        # AP@100
                                        EvalFunction.eval(
                                            [
                                                "-c",
                                                "-m",
                                                "map_cut.100",
                                                "-l2",
                                                TOPICS[dataset],
                                                f,
                                            ]
                                        ),
                                        # R@20
                                        EvalFunction.eval(
                                            [
                                                "-c",
                                                "-m",
                                                "recall.20",
                                                TOPICS[dataset],
                                                f,
                                            ]
                                        ),
                                        # R@100
                                        EvalFunction.eval(
                                            [
                                                "-c",
                                                "-m",
                                                "recall.100",
                                                TOPICS[dataset],
                                                f,
                                            ]
                                        ),
                                    ],
                                },
                                output,
                            )
                            output.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
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
        "--prompt_mode",
        type=PromptMode,
        required=True,
        choices=list(PromptMode),
    )
    parser.add_argument(
        "--rerank_results_dirname",
        type=str,
        default="rerank_results",
        help="name of the directory used for storing rerank results",
    )
    args = parser.parse_args()
    main(args)

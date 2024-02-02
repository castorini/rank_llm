import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod
from rank_llm.retrieve.topics_dict import TOPICS


def main(args):
    # TODO: make metrics configurable
    model = args.model_name
    context_size = args.context_size
    prompt_mode = args.prompt_mode
    output_filename = f"trec_eval_aggregated_results_{model}_{prompt_mode}.jsonl"
    with open(output_filename, "w") as output:
        for dataset in ["dl19", "dl20", "dl21", "dl22", "news", "covid"]:
            for retrieval_method in RetrievalMethod:
                if retrieval_method == RetrievalMethod.UNSPECIFIED:
                    continue
                for top_k_canidadates in [20, 100]:
                    directory = f"rerank_results/{retrieval_method.name}"
                    for filename in os.listdir(directory):
                        if not filename.startswith(
                            f"{model}_{context_size}_{top_k_canidadates}_{prompt_mode}_{dataset}"
                        ):
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

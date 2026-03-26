from __future__ import annotations

import json
import os
from argparse import Namespace
from collections.abc import Sequence

from rank_llm.cli.legacy import namespace_to_legacy_argv, translate_legacy_argv
from rank_llm.cli.main import main as cli_main
from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.retrieve import TOPICS, RetrievalMethod


def evaluate_aggregate(args: Namespace) -> None:
    model = args.model_name
    context_size = args.context_size
    rerank_results_dirname = args.rerank_results_dirname
    output_filename = f"trec_eval_aggregated_results_{model}.jsonl"
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
                            f"{model}_{context_size}_{top_k_canidadates}_{dataset}"
                        ):
                            continue
                        if filename.endswith(".json"):
                            continue
                        file_path = os.path.join(directory, filename)
                        if os.path.isfile(file_path):
                            json.dump(
                                {
                                    "file": file_path,
                                    "result": [
                                        EvalFunction.eval(
                                            [
                                                "-c",
                                                "-m",
                                                "ndcg_cut.10",
                                                TOPICS[dataset],
                                                file_path,
                                            ]
                                        ),
                                        EvalFunction.eval(
                                            [
                                                "-c",
                                                "-m",
                                                "map_cut.100",
                                                "-l2",
                                                TOPICS[dataset],
                                                file_path,
                                            ]
                                        ),
                                        EvalFunction.eval(
                                            [
                                                "-c",
                                                "-m",
                                                "recall.20",
                                                TOPICS[dataset],
                                                file_path,
                                            ]
                                        ),
                                        EvalFunction.eval(
                                            [
                                                "-c",
                                                "-m",
                                                "recall.100",
                                                TOPICS[dataset],
                                                file_path,
                                            ]
                                        ),
                                    ],
                                },
                                output,
                            )
                            output.write("\n")


def main(args: Namespace | Sequence[str] | None = None) -> int:
    if isinstance(args, Namespace):
        argv = namespace_to_legacy_argv(args)
    elif args is None:
        import sys

        argv = sys.argv[1:]
    else:
        argv = list(args)

    translated = translate_legacy_argv(argv)
    return cli_main(["evaluate", *translated])


if __name__ == "__main__":
    raise SystemExit(main())

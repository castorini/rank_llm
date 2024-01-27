from argparse import ArgumentParser
import json
import os
import platform
import pandas as pd
import subprocess
import tempfile

from pyserini.search import get_qrels_file
from pyserini.util import download_evaluation_script

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from rank_llm.rerank.rankllm import PromptMode
from rank_llm.retrieve.pyserini_retriever import RetrievalMethod
from rank_llm.retrieve.topics_dict import TOPICS


class EvalFunction:
    @staticmethod
    def trunc(qrels, run):
        qrels = get_qrels_file(qrels)
        run = pd.read_csv(run, sep="\s+", header=None)
        qrels = pd.read_csv(qrels, sep="\s+", header=None)
        run[0] = run[0].astype(str)
        qrels[0] = qrels[0].astype(str)

        qrels = qrels[qrels[0].isin(run[0])]
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        qrels.to_csv(temp_file, sep="\t", header=None, index=None)
        return temp_file

    @staticmethod
    def eval(args, trunc=True):
        script_path = download_evaluation_script("trec_eval")
        cmd_prefix = ["java", "-jar", script_path]
        # args = sys.argv

        # Option to discard non-judged hits in run file
        judged_docs_only = ""
        judged_result = []
        cutoffs = []

        if "-remove-unjudged" in args:
            judged_docs_only = args.pop(args.index("-remove-unjudged"))

        if any([i.startswith("judged.") for i in args]):
            # Find what position the arg is in.
            idx = [i.startswith("judged.") for i in args].index(True)
            cutoffs = args.pop(idx)
            cutoffs = list(map(int, cutoffs[7:].split(",")))
            # Get rid of the '-m' before the 'judged.xxx' option
            args.pop(idx - 1)

        temp_file = ""

        if len(args) > 1:
            if trunc:
                args[-2] = EvalFunction.trunc(args[-2], args[-1])
                print("Trunc", args[-2])

            if not os.path.exists(args[-2]):
                args[-2] = get_qrels_file(args[-2])
            if os.path.exists(args[-1]):
                # Convert run to trec if it's on msmarco
                with open(args[-1]) as f:
                    first_line = f.readline()
                if "Q0" not in first_line:
                    temp_file = tempfile.NamedTemporaryFile(delete=False).name
                    print("msmarco run detected. Converting to trec...")
                    run = pd.read_csv(
                        args[-1],
                        sep="\s+",
                        header=None,
                        names=["query_id", "doc_id", "rank"],
                    )
                    run["score"] = 1 / run["rank"]
                    run.insert(1, "Q0", "Q0")
                    run["name"] = "TEMPRUN"
                    run.to_csv(temp_file, sep="\t", header=None, index=None)
                    args[-1] = temp_file

            run = pd.read_csv(args[-1], sep="\s+", header=None)
            qrels = pd.read_csv(args[-2], sep="\s+", header=None)

            # cast doc_id column as string
            run[0] = run[0].astype(str)
            qrels[0] = qrels[0].astype(str)

            # Discard non-judged hits

            if judged_docs_only:
                if not temp_file:
                    temp_file = tempfile.NamedTemporaryFile(delete=False).name
                judged_indexes = pd.merge(
                    run[[0, 2]].reset_index(), qrels[[0, 2]], on=[0, 2]
                )["index"]
                run = run.loc[judged_indexes]
                run.to_csv(temp_file, sep="\t", header=None, index=None)
                args[-1] = temp_file
            # Measure judged@cutoffs
            for cutoff in cutoffs:
                run_cutoff = run.groupby(0).head(cutoff)
                judged = len(
                    pd.merge(run_cutoff[[0, 2]], qrels[[0, 2]], on=[0, 2])
                ) / len(run_cutoff)
                metric_name = f"judged_{cutoff}"
                judged_result.append(f"{metric_name:22}\tall\t{judged:.4f}")
            cmd = cmd_prefix + args[1:]
        else:
            cmd = cmd_prefix

        print(f"Running command: {cmd}")
        shell = platform.system() == "Windows"
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
        )
        stdout, stderr = process.communicate()
        if stderr:
            print(stderr.decode("utf-8"))

        print("Results:")
        results = stdout.decode("utf-8").rstrip()
        print(results)

        for judged in judged_result:
            print(judged)

        if temp_file:
            os.remove(temp_file)
        return results


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
                directory = f"rerank_results/{retrieval_method.name}"
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
    args = parser.parse_args()
    main(args)

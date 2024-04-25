import os
import platform
import subprocess
import tempfile
from typing import List

import pandas as pd
from pyserini.search import get_qrels_file
from pyserini.util import download_evaluation_script

from rank_llm.data import Result


class EvalFunction:
    @staticmethod
    def from_results(
        results: List[Result],
        qrels: str,
        eval_args: list[str] = ["-c", "-m", "ndcg_cut.10"],
    ) -> str:
        """
        This method processes a list of Result objects and immediately evaluates them,
        returning the evaluation result as a string.

        Args:
            results (List[Result]): A list of Result objects.
            qrels (str): Path to the qrels file.

        Returns:
            str: Evaluation results as a string.
        """
        # Convert the list of Result objects to a temporary run file format
        temp_run_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".txt"
        ).name
        with open(temp_run_file, "w") as file:
            for result in results:
                qid = result.query.qid
                for rank, cand in enumerate(result.candidates, start=1):
                    file.write(f"{qid} Q0 {cand.docid} {rank} {cand.score} rank_llm\n")
        # make a deep copy of eval_args to preserve its default value for the next
        args = []
        args.extend(eval_args)
        args.append(qrels)
        args.append(temp_run_file)
        eval_result = EvalFunction.eval(args, trunc=True)
        os.remove(temp_run_file)

        return eval_result

    @staticmethod
    def trunc(qrels: str, run: str):
        """
        Truncates the qrels file to only include queries that are present in the given run file.

        Args:
            qrels (str): Path to the qrels file.
            run (str): Path to the run file.

        Returns:
            str: Path to the truncated qrels file.
        """
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
        """
        Runs the evaluation script with the given list of arguments and options.

        Args:
            args (list): Arguments to be passed to the evaluation script.
            trunc (bool, optional): Indicates whether to truncate qrels file. Defaults to True.

        Returns:
            str: Evaluation results as a string.
        """
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

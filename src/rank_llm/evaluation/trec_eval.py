import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from rank_llm._optional import missing_extra_error

try:
    from pyserini.search import get_qrels_file
except ImportError:
    get_qrels_file = None

from rank_llm.data import Result


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as err:
        raise missing_extra_error("pyserini", "Evaluation requires pandas.") from err
    return pd


def _require_trec_eval():
    try:
        from pyserini.eval.trec_eval import trec_eval
    except ImportError as err:
        raise missing_extra_error("pyserini", "Evaluation requires Pyserini.") from err
    return trec_eval


class EvalFunction:
    @staticmethod
    def from_trec_runfile(
        run_file: str,
        qrels: str,
        eval_args: list[str] | None = None,
    ) -> str:
        """
        This method processes a trec run file and evaluates it returning the evaluation result as a string.

        Args:
            run_file: A tsv file with results stored in trec format.
            qrels (str): Path to the qrels file.

        Returns:
            str: Evaluation results as a string.
        """

        # make a deep copy of eval_args to preserve its default value for the next run
        args = []
        if eval_args is None:
            eval_args = ["-c", "-m", "ndcg_cut.10"]
        args.extend(eval_args)
        args.append(qrels)
        args.append(run_file)
        eval_result = EvalFunction.eval(args, trunc=True)

        return eval_result

    @staticmethod
    def from_results(
        results: list[Result],
        qrels: str,
        eval_args: list[str] | None = None,
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
        # make a deep copy of eval_args to preserve its default value for the next run
        args = []
        if eval_args is None:
            eval_args = ["-c", "-m", "ndcg_cut.10"]
        args.extend(eval_args)
        args.append(qrels)
        args.append(temp_run_file)
        try:
            return EvalFunction.eval(args, trunc=True)
        finally:
            os.remove(temp_run_file)

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
        if not Path(qrels).exists():
            if get_qrels_file is None:
                raise missing_extra_error(
                    "pyserini",
                    "Evaluation requires Pyserini and pandas-backed qrels loading.",
                )
            qrels = get_qrels_file(qrels)

        pd = _require_pandas()
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
        trec_eval = _require_trec_eval()
        args = list(args)
        truncated_qrels = None
        output = StringIO()
        try:
            if trunc and len(args) > 1:
                truncated_qrels = EvalFunction.trunc(args[-2], args[-1])
                args[-2] = truncated_qrels
            # Pyserini expects argv[0] and prints the full metric output. Its return
            # value only represents the last metric, so preserve stdout instead.
            with redirect_stdout(output):
                trec_eval(["trec_eval", *args], return_per_query_results="-q" in args)
        finally:
            if truncated_qrels is not None:
                os.remove(truncated_qrels)
            text = output.getvalue().rstrip()
            if text:
                print(text)
        return text

import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO

from rank_llm._optional import missing_extra_error
from rank_llm.data import Result


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
        eval_result = EvalFunction.eval(args)

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
            return EvalFunction.eval(args)
        finally:
            os.remove(temp_run_file)

    @staticmethod
    def eval(args):
        """
        Runs the evaluation script with the given list of arguments and options.

        Args:
            args (list): Arguments to be passed to the evaluation script.

        Returns:
            str: Evaluation results as a string.
        """
        trec_eval = _require_trec_eval()
        args = list(args)
        output = StringIO()
        try:
            # Pyserini expects argv[0] and prints the full metric output. Its return
            # value only represents the last metric, so preserve stdout instead.
            with redirect_stdout(output):
                trec_eval(["trec_eval", *args], return_per_query_results="-q" in args)
        finally:
            text = output.getvalue().rstrip()
            if text:
                print(text)
        return text

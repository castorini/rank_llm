#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import glob
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from string import Template

import pkg_resources
import yaml

from ._base import fail_str, ok_str, okish_str, run_eval_and_return_metric

dense_threads = 16
dense_batch_size = 512
sparse_threads = 16
sparse_batch_size = 128

# The models: the rows of the results table will be ordered this way.
models = {
    # MS MARCO v1 passage
    "msmarco-v1-passage": [
        "rank_vicuna_7b_v1_splade_pp",
        "rank_zephyr_7b_v1_full_splade_pp",
        "rank_zephyr_7b_v1_full_splade_pp_mult_pass",
        "monot5",
        "duot5",
        "lit5distill",
        "rank_vicuna_7b_v1_bm25",
        "rank_zephyr_7b_v1_full_bm25",
        "firstmistral",
        "qwen_2.5_7b_inst",
        "llama_3.1_8b_inst",
        "gemini_flash_2.0",
        "rankgpt",
        "rankgptapeer",
        "lrl",
    ],
    # MS MARCO v2 passage
    "msmarco-v2-passage": [
        "monot5",
        "duot5",
        "lit5distill",
        "rankvicuna",
        "rankzephyr",
        "firstmistral",
        "qwen_2.5_7b_inst",
        "llama_3.1_8b_inst",
        "gemini_2.0_flash",
        "rankgpt",
        "rankgptapeer",
        "lrl",
    ],
    #  ,
    # # MS MARCO v1 doc
    # 'msmarco-v1-doc':
    # ['rank_zephyr_7b_v1_full',
    # 'rank_vicuna_7b_v1'],
    # # MS MARCO v2 passage
    # 'msmarco-v2-passage':
    # ['rank_zephyr_7b_v1_full',
    # 'rank_vicuna_7b_v1'],
    # 'msmarco-v2-doc':
    # ['rank_zephyr_7b_v1_full',
    # 'rank_vicuna_7b_v1'
    #  ]
}

trec_eval_metric_definitions = {
    "msmarco-v1-passage": {
        "msmarco-passage-dev-subset": {
            "MRR@10": "-c -M 10 -m recip_rank",
            "R@1K": "-c -m recall.1000",
        },
        "dl19-passage": {
            # 'MAP': '-c -l 2 -m map',
            "nDCG@10": "-c -m ndcg_cut.10",
            # 'R@1K': '-c -l 2 -m recall.1000'
        },
        "dl20-passage": {
            # 'MAP': '-c -l 2 -m map',
            "nDCG@10": "-c -m ndcg_cut.10",
            # 'R@1K': '-c -l 2 -m recall.1000'
        },
    },
    "msmarco-v2-passage": {
        "dl21-passage": {
            # 'MAP': '-c -l 2 -m map',
            "nDCG@10": "-c -m ndcg_cut.10",
            # 'R@1K': '-c -l 2 -m recall.1000'
        },
        "dl22-passage": {
            # 'MAP': '-c -l 2 -m map',
            "nDCG@10": "-c -m ndcg_cut.10",
            # 'R@1K': '-c -l 2 -m recall.1000'
        },
        "dl23-passage": {
            # 'MAP': '-c -l 2 -m map',
            "nDCG@10": "-c -m ndcg_cut.10",
            # 'R@1K': '-c -l 2 -m recall.1000'
        },
    },
}


def find_msmarco_table_topic_set_key_v1(topic_key):
    # E.g., we want to map variants like 'dl19-passage-unicoil' and 'dl19-passage' both into 'dl19'
    key = ""
    if topic_key.startswith("dl19"):
        key = "dl19"
    elif topic_key.startswith("dl20"):
        key = "dl20"
    elif topic_key.startswith("msmarco"):
        key = "dev"

    return key


def find_msmarco_table_topic_set_key_v2(topic_key):
    key = ""
    if (
        topic_key.endswith("dev")
        or topic_key.endswith("dev-unicoil")
        or topic_key.endswith("dev-unicoil-noexp")
    ):
        key = "dev"
    elif (
        topic_key.endswith("dev2")
        or topic_key.endswith("dev2-unicoil")
        or topic_key.endswith("dev2-unicoil-noexp")
    ):
        key = "dev2"
    elif topic_key.startswith("dl21"):
        key = "dl21"
    elif topic_key.startswith("dl22"):
        key = "dl22"
    elif topic_key.startswith("dl23"):
        key = "dl23"

    return key


def format_eval_command(raw):
    return raw.replace("run.", "\\\n  run.").replace("key", "\\\n  key")


def format_command(raw):
    # Format hybrid commands differently.
    if "pyserini.search.hybrid" in raw:
        return (
            raw.replace("dense", "\\\n  dense ")
            .replace("--encoder", "\\\n         --encoder")
            .replace("sparse", "\\\n  sparse")
            .replace("fusion", "\\\n  fusion")
            .replace("run --", "\\\n  run    --")
            .replace("--topics ", "\\\n         --topics ")
            .replace("--output ", "\\\n         --output ")
        )

    # After "--output foo.txt" are additional options like "--hits 1000 --impact".
    # We want these on a separate line for better readability, but note that sometimes that might
    # be the end of the command, in which case we don't want to add an extra line break.
    return (
        raw.replace("--topics", "\\\n  --topics")
        .replace("--threads", "\\\n  --threads")
        .replace("--index", "\\\n  --index")
        .replace("--output ", "\\\n  --output ")
        .replace("--encoder", "\\\n  --encoder")
        .replace("--onnx-encoder", "\\\n  --onnx-encoder")
        .replace("--encoded-corpus", "\\\n  --encoded-corpus")
        .replace("--context_size", "\\\n  --context_size")
        .replace("--prompt_mode", "\\\n  --prompt_mode")
        .replace("--top_k_candidates", "\\\n  --top_k_candidates")
        .replace("--retrieval_method", "\\\n  --retrieval_method")
        .replace("--model_path", "\\\n  --model_path")
        .replace("--variable_passage", "\\\n  --variable_passage")
        .replace("--num_passes", "\\\n  --num_passes")
        .replace(".txt ", ".txt \\\n  ")
    )


def read_file(f):
    fin = open(f, "r")
    text = fin.read()
    fin.close()

    return text


def list_conditions(args):
    for condition in models[args.collection]:
        if condition == "":
            continue
        print(condition)


def generate_report(args):
    yaml_file = pkg_resources.resource_filename(__name__, f"{args.collection}.yaml")

    version = "v1" if "v1" in args.collection else "v2"
    corpus_type = "passage" if "passage" in args.collection else "doc"

    html_template = read_file(
        pkg_resources.resource_filename(
            __name__, f"msmarco_html_{version}_{corpus_type}.template"
        )
    )
    row_template = read_file(
        pkg_resources.resource_filename(
            __name__, f"msmarco_html_row_{version}.template"
        )
    )

    table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
    commands = defaultdict(lambda: defaultdict(lambda: ""))
    eval_commands = defaultdict(lambda: defaultdict(lambda: ""))
    table_keys = {}
    row_ids = {}

    with open(yaml_file) as f:
        yaml_data = yaml.safe_load(f)
        for condition in yaml_data["conditions"]:
            name = condition["name"]
            table_keys[name] = condition["display-html"]
            row_ids[name] = condition.get("display-row", "")

            for topic_set in condition["topics"]:
                topic_key = topic_set["topic_key"]
                short_topic_key = (
                    find_msmarco_table_topic_set_key_v1(topic_key)
                    if version == "v1"
                    else find_msmarco_table_topic_set_key_v2(topic_key)
                )

                latest_file = max(
                    glob.glob(
                        os.path.join(
                            "rerank_results/SPLADE_P_P_ENSEMBLE_DISTIL/",
                            f"{name}_4096_100_rank_GPT_{short_topic_key}*.txt",
                        )
                    ),
                    key=os.path.getctime,
                    default=None,
                )
                runfile = f'"rerank_results/SPLADE_P_P_ENSEMBLE_DISTIL/{name}_4096_100_rank_GPT_{short_topic_key}*.txt\\")'
                cmd = Template(condition["command"]).substitute(
                    topics=topic_key,
                    output=runfile,
                    sparse_threads=sparse_threads,
                    sparse_batch_size=sparse_batch_size,
                    dense_threads=dense_threads,
                    dense_batch_size=dense_batch_size,
                )
                commands[name][short_topic_key] = cmd

                for expected in topic_set["scores"]:
                    for metric in expected:
                        table[name][short_topic_key][metric] = expected[metric]

    html_rows = []
    row_cnt = 1

    for name in models[args.collection]:
        if not name:
            html_rows.append('<tr><td style="border-bottom: 0"></td></tr>')
            continue

        row_data = {
            "row_cnt": row_cnt,
            "condition_name": table_keys[name],
            "row": row_ids[name],
        }

        if version == "v1":
            row_data.update(
                {
                    "s1": f'{table[name]["dl19"]["MULT"]:.0f}'
                    if table[name]["dl19"]["MULT"] != 0
                    else "-",
                    "s2": "SPLADE++ EnsembleDistil"
                    if "SPLADE++_EnsembleDistil_ONNX" in commands[name]["dl19"]
                    else "BM25",
                    "s3": re.search(
                        r"--top_k_candidates=(\d+)", commands[name]["dl19"]
                    ).group(1)
                    if re.search(r"--top_k_candidates=(\d+)", commands[name]["dl19"])
                    else "-",
                    "s4": f'{table[name]["dl19"]["nDCG@10"]:.4f}'
                    if table[name]["dl19"]["nDCG@10"] != 0
                    else "-",
                    "s5": f'{table[name]["dl20"]["nDCG@10"]:.4f}'
                    if table[name]["dl20"]["nDCG@10"] != 0
                    else "-",
                    "cmd1": format_command(commands[name]["dl19"]),
                    "cmd2": format_command(commands[name]["dl20"]),
                    "eval_cmd1": format_eval_command(eval_commands[name]["dl19"]),
                    "eval_cmd2": format_eval_command(eval_commands[name]["dl20"]),
                }
            )
        else:
            row_data.update(
                {
                    "s1": f'{table[name]["dl21"]["MULT"]:.0f}'
                    if table[name]["dl21"]["MULT"] != 0
                    else "-",
                    "s2": "SPLADE++ EnsembleDistil"
                    if "SPLADE++_EnsembleDistil_ONNX" in commands[name]["dl21"]
                    else "BM25",
                    "s3": re.search(
                        r"--top_k_candidates=(\d+)", commands[name]["dl21"]
                    ).group(1)
                    if re.search(r"--top_k_candidates=(\d+)", commands[name]["dl21"])
                    else "-",
                    "s4": f'{table[name]["dl21"].get("nDCG@10", 0):.4f}'
                    if "dl21" in table[name]
                    else "-",
                    "s5": f'{table[name]["dl22"].get("nDCG@10", 0):.4f}'
                    if "dl22" in table[name]
                    else "-",
                    "s6": f'{table[name]["dl23"].get("nDCG@10", 0):.4f}'
                    if "dl23" in table[name]
                    else "-",
                    "cmd1": format_command(commands[name]["dl21"]),
                    "cmd2": format_command(commands[name]["dl22"]),
                    "cmd3": format_command(commands[name]["dl23"]),
                    "eval_cmd1": format_eval_command(eval_commands[name]["dl21"]),
                    "eval_cmd2": format_eval_command(eval_commands[name]["dl22"]),
                    "eval_cmd3": format_eval_command(eval_commands[name]["dl23"]),
                }
            )

        html_rows.append(Template(row_template).substitute(**row_data))
        row_cnt += 1

    full_name = f"MS MARCO {version.upper()} {corpus_type.capitalize()}"
    with open(args.output, "w") as out:
        out.write(
            Template(html_template).substitute(
                title=full_name, rows="\n".join(html_rows)
            )
        )


def run_conditions(args):
    start = time.time()

    table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
    table_keys = {}

    yaml_file = pkg_resources.resource_filename(__name__, f"{args.collection}.yaml")

    with open(yaml_file) as f:
        yaml_data = yaml.safe_load(f)
        for condition in yaml_data["conditions"]:
            # Either we're running all conditions, or running only the condition specified in --condition
            if not args.all:
                if not condition["name"] == args.condition:
                    continue

            name = condition["name"]
            display = condition["display"]
            cmd_template = condition["command"]
            print(f"file: {yaml_file}")

            print(f'# Running condition "{name}": {display}\n')
            for topic_set in condition["topics"]:
                topic_key = topic_set["topic_key"]
                eval_key = topic_set["eval_key"]

                if (
                    args.collection == "msmarco-v1-passage"
                    or args.collection == "msmarco-v1-doc"
                ):
                    short_topic_key = find_msmarco_table_topic_set_key_v1(topic_key)
                else:
                    short_topic_key = find_msmarco_table_topic_set_key_v2(topic_key)

                print(f"  - topic_key: {topic_key}")

                runfile = os.path.join(
                    args.directory,
                    f"run.{args.collection}.{name}.{short_topic_key}.txt",
                )
                cmd = Template(cmd_template).substitute(
                    topics=topic_key,
                    output=runfile,
                    sparse_threads=sparse_threads,
                    sparse_batch_size=sparse_batch_size,
                    dense_threads=dense_threads,
                    dense_batch_size=dense_batch_size,
                )
                if args.display_commands:
                    print(f"\n```bash\n{format_command(cmd)}\n```\n")

                if not os.path.exists(runfile):
                    if not args.dry_run:
                        os.system(cmd)

                for expected in topic_set["scores"]:
                    for metric in expected:
                        table_keys[name] = display
                        if not args.skip_eval:
                            # If the runfile doesn't exist, we can't evaluate.
                            # This would be the case if --dry-run were set.
                            if not os.path.exists(runfile):
                                continue

                            score = float(
                                run_eval_and_return_metric(
                                    metric,
                                    eval_key,
                                    trec_eval_metric_definitions[args.collection][
                                        eval_key
                                    ][metric],
                                    runfile,
                                )
                            )
                            if math.isclose(score, float(expected[metric])):
                                result_str = ok_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "distilbert-kd-tasb-rocchio-prf-pytorch"
                                and topic_key == "msmarco-passage-dev-subset"
                                and metric == "MRR@10"
                                and abs(score - float(expected[metric])) <= 0.0001
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "tct_colbert-v2-hnp-avg-prf-pytorch"
                                and topic_key == "dl19-passage"
                                and metric == "nDCG@10"
                                and abs(score - float(expected[metric])) <= 0.0001
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "tct_colbert-v2-hnp-avg-prf-pytorch"
                                and topic_key == "dl20"
                                and metric == "MAP"
                                and abs(score - float(expected[metric])) <= 0.0002
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "tct_colbert-v2-hnp-avg-prf-pytorch"
                                and topic_key == "dl20"
                                and metric == "nDCG@10"
                                and abs(score - float(expected[metric])) <= 0.0009
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "tct_colbert-v2-hnp-bm25-pytorch"
                                and topic_key == "msmarco-passage-dev-subset"
                                and metric == "MRR@10"
                                and abs(score - float(expected[metric])) <= 0.0001
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "ance"
                                and topic_key == "msmarco-passage-dev-subset"
                                and metric == "MRR@10"
                                and abs(score - float(expected[metric])) <= 0.0001
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "ance-pytorch"
                                and topic_key == "msmarco-passage-dev-subset"
                                and metric == "MRR@10"
                                and abs(score - float(expected[metric])) <= 0.0001
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "ance-rocchio-prf-pytorch"
                                and topic_key == "msmarco-passage-dev-subset"
                                and metric == "R@1K"
                                and abs(score - float(expected[metric])) <= 0.0002
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "ance-rocchio-prf-pytorch"
                                and topic_key == "dl19-passage"
                                and metric == "MAP"
                                and abs(score - float(expected[metric])) <= 0.0001
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "ance-rocchio-prf-pytorch"
                                and topic_key == "dl19-passage"
                                and metric == "nDCG@10"
                                and abs(score - float(expected[metric])) <= 0.0008
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "ance-rocchio-prf-pytorch"
                                and topic_key == "dl19-passage"
                                and metric == "nDCG@10"
                                and abs(score - float(expected[metric])) <= 0.0007
                            ):
                                result_str = okish_str
                            # Flaky test on Jimmy's Mac Studio
                            elif (
                                args.collection == "msmarco-v1-passage"
                                and name == "ance-avg-prf-pytorch"
                                and topic_key == "msmarco-passage-dev-subset"
                                and metric == "MRR@10"
                                and abs(score - float(expected[metric])) <= 0.0002
                            ):
                                result_str = okish_str
                            else:
                                result_str = (
                                    fail_str + f" expected {expected[metric]:.4f}"
                                )
                            print(f"    {metric:7}: {score:.4f} {result_str}")
                            table[name][short_topic_key][metric] = score
                        else:
                            table[name][short_topic_key][metric] = expected[metric]

                if not args.skip_eval:
                    print("")

    if args.collection == "msmarco-v1-passage" or args.collection == "msmarco-v1-doc":
        print(
            " " * 74 + "TREC 2019" + " " * 16 + "TREC 2020" + " " * 12 + "MS MARCO dev"
        )
        print(
            " " * 67
            + "MAP    nDCG@10    R@1K       MAP nDCG@10    R@1K    MRR@10    R@1K"
        )
        print(" " * 67 + "-" * 22 + "    " + "-" * 22 + "    " + "-" * 14)

        if args.condition:
            # If we've used --condition to specify a specific condition, print out only that row.
            name = args.condition
            print(
                f"{table_keys[name]:65}"
                + f'{table[name]["dl19"]["MAP"]:8.4f}{table[name]["dl19"]["nDCG@10"]:8.4f}{table[name]["dl19"]["R@1K"]:8.4f}  '
                + f'{table[name]["dl20"]["MAP"]:8.4f}{table[name]["dl20"]["nDCG@10"]:8.4f}{table[name]["dl20"]["R@1K"]:8.4f}  '
                + f'{table[name]["dev"]["MRR@10"]:8.4f}{table[name]["dev"]["R@1K"]:8.4f}'
            )
        else:
            # Otherwise, print out all rows
            for name in models[args.collection]:
                if not name:
                    print("")
                    continue
                print(
                    f"{table_keys[name]:65}"
                    + f'{table[name]["dl19"]["MAP"]:8.4f}{table[name]["dl19"]["nDCG@10"]:8.4f}{table[name]["dl19"]["R@1K"]:8.4f}  '
                    + f'{table[name]["dl20"]["MAP"]:8.4f}{table[name]["dl20"]["nDCG@10"]:8.4f}{table[name]["dl20"]["R@1K"]:8.4f}  '
                    + f'{table[name]["dev"]["MRR@10"]:8.4f}{table[name]["dev"]["R@1K"]:8.4f}'
                )
    elif args.collection == "msmarco-v2-passage":
        print(" " * 50 + "TREC 2021" + " " * 14 + "TREC 2022" + " " * 14 + "TREC 2023")
        print(" " * 50 + "nDCG@10" + " " * 14 + "nDCG@10" + " " * 14 + "nDCG@10")
        print(" " * 50 + "-" * 8 + " " * 14 + "-" * 8 + " " * 14 + "-" * 8)

        if args.condition:
            name = args.condition
            print(
                f"{table_keys[name]:48}"
                + f'{table[name]["dl21"]["nDCG@10"]:8.4f}'
                + " " * 14
                + f'{table[name]["dl22"]["nDCG@10"]:8.4f}'
                + " " * 14
                + f'{table[name]["dl23"]["nDCG@10"]:8.4f}'
            )
        else:
            for name in models[args.collection]:
                if not name:
                    print("")
                    continue
                print(
                    f"{table_keys[name]:48}"
                    + f'{table[name]["dl21"]["nDCG@10"]:8.4f}'
                    + " " * 14
                    + f'{table[name]["dl22"]["nDCG@10"]:8.4f}'
                    + " " * 14
                    + f'{table[name]["dl23"]["nDCG@10"]:8.4f}'
                )
    else:
        print(
            " " * 77
            + "TREC 2021"
            + " " * 18
            + "MS MARCO dev"
            + " " * 6
            + "MS MARCO dev2"
        )
        print(
            " " * 62
            + "MAP@100 nDCG@10 MRR@100 R@100   R@1K     MRR@100   R@1K    MRR@100   R@1K"
        )
        print(" " * 62 + "-" * 38 + "    " + "-" * 14 + "    " + "-" * 14)

        if args.condition:
            # If we've used --condition to specify a specific condition, print out only that row.
            name = args.condition
            print(
                f"{table_keys[name]:60}"
                + f'{table[name]["dl21"]["MAP@100"]:8.4f}{table[name]["dl21"]["nDCG@10"]:8.4f}'
                + f'{table[name]["dl21"]["MRR@100"]:8.4f}{table[name]["dl21"]["R@100"]:8.4f}{table[name]["dl21"]["R@1K"]:8.4f}  '
                + f'{table[name]["dev"]["MRR@100"]:8.4f}{table[name]["dev"]["R@1K"]:8.4f}  '
                + f'{table[name]["dev2"]["MRR@100"]:8.4f}{table[name]["dev2"]["R@1K"]:8.4f}'
            )
        else:
            # Otherwise, print out all rows
            for name in models[args.collection]:
                if not name:
                    print("")
                    continue
                print(
                    f"{table_keys[name]:60}"
                    + f'{table[name]["dl21"]["MAP@100"]:8.4f}{table[name]["dl21"]["nDCG@10"]:8.4f}'
                    + f'{table[name]["dl21"]["MRR@100"]:8.4f}{table[name]["dl21"]["R@100"]:8.4f}{table[name]["dl21"]["R@1K"]:8.4f}  '
                    + f'{table[name]["dev"]["MRR@100"]:8.4f}{table[name]["dev"]["R@1K"]:8.4f}  '
                    + f'{table[name]["dev2"]["MRR@100"]:8.4f}{table[name]["dev2"]["R@1K"]:8.4f}'
                )

    end = time.time()
    start_str = datetime.utcfromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S")
    end_str = datetime.utcfromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S")

    print("\n")
    print(f"Start time: {start_str}")
    print(f"End time: {end_str}")
    print(f"Total elapsed time: {end - start:.0f}s ~{(end - start)/3600:.1f}hr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate regression matrix for MS MARCO corpora."
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Collection = {v1-passage, v1-doc, v2-passage, v2-doc}.",
        required=True,
    )
    # To list all conditions
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        default=False,
        help="List available conditions.",
    )
    # For generating reports
    parser.add_argument(
        "--generate-report", action="store_true", default=False, help="Generate report."
    )
    parser.add_argument(
        "--output", type=str, help="File to store report.", required=False
    )
    # For actually running the experimental conditions
    parser.add_argument(
        "--all", action="store_true", default=False, help="Run all conditions."
    )
    parser.add_argument(
        "--condition", type=str, help="Condition to run.", required=False
    )
    parser.add_argument(
        "--directory", type=str, help="Base directory.", default="", required=False
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print out commands but do not execute.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Skip running trec_eval.",
    )
    parser.add_argument(
        "--display-commands",
        action="store_true",
        default=False,
        help="Display command.",
    )
    args = parser.parse_args()

    if args.collection == "v1-passage":
        args.collection = "msmarco-v1-passage"
    elif args.collection == "v1-doc":
        args.collection = "msmarco-v1-doc"
    elif args.collection == "v2-passage":
        args.collection = "msmarco-v2-passage"
    elif args.collection == "v2-doc":
        args.collection = "msmarco-v2-doc"
    else:
        raise ValueError(f"Unknown corpus: {args.collection}")

    if args.list_conditions:
        list_conditions(args)
        sys.exit()

    if args.generate_report:
        if not args.output:
            print(f"Must specify report filename with --output.")
            sys.exit()

        generate_report(args)
        sys.exit()

    if not args.all and not args.condition:
        print(
            f"Must specify a specific condition using --condition or use --all to run all conditions."
        )
        sys.exit()

    run_conditions(args)

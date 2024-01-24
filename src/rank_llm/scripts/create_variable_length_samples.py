import argparse
import json
import random
import re


def create_prompt(passage_list, selected_indexes, query):
    num_selected_passages = len(selected_indexes)
    prefix = (
        f"I will provide you with {num_selected_passages} passages, each indicated by a numerical identifier []. "
        f"Rank the passages based on their relevance to the search query: {query}\n\n"
    )
    passages = ""
    for i, v in enumerate(selected_indexes):
        passages += f"[{i+1}] {passage_list[v - 1]}\n"
    suffix = (
        f"Search Query: {query}\nRank the {num_selected_passages} passages above based on their relevance "
        "to the search query. All the passages should be included and listed using identifiers, "
        "in descending order of relevance. The output format should be [] > [], e.g., [2] > [1], "
        "Only respond with the ranking results, do not say any word or explain."
    )
    return prefix + passages + suffix


def read_data(data_path):
    with open(data_path, "r", encoding="utf-8") as json_file:
        json_objects = json.load(json_file)
    return json_objects


def sample_data(json_objects, num_passages, num_samples, include_original):
    new_objs = []
    num_selected_passages = []
    num_passage_stats = {}
    for i in range(2, num_passages + 1):
        num_passage_stats[i] = 0
    rank_frequencies = {}
    for i in range(num_passages):
        rank_frequencies[i + 1] = 0
    skipped = []
    for obj in json_objects:
        prompt_index = 1 if len(obj["conversations"]) == 3 else 0
        response_index = 2 if len(obj["conversations"]) == 3 else 1
        # print("json_obj:\t", obj)
        prompt = obj["conversations"][prompt_index]["value"]
        q_b_index = prompt.rfind("\nSearch Query:")
        q_e_index = prompt.rfind("\nRank the 20 passages above based on their")
        query = prompt[q_b_index + len("\nSearch Query:") + 1 : q_e_index]
        prompt = prompt[:q_b_index]
        # print("query:\t", query)
        passages = re.split(r"\n\[\d+\] ", prompt)[1:]
        if num_passages != len(passages):
            print(len(passages))
            print(obj)
            skipped.append(obj)
            continue
        assert num_passages == len(passages)
        response = obj["conversations"][response_index]["value"][1:-1]
        rankings = {}
        for i, s in enumerate(response.split("] > [")):
            rankings[int(s)] = i + 1
        assert num_passages == len(rankings)
        # print(rankings)
        # print(response)
        if include_original:
            obj["conversations"][prompt_index]["value"] = obj["conversations"][
                prompt_index
            ]["value"].replace(f"e.g., [4] > [2], ", f"e.g., [2] > [1], ")
            new_objs.append(obj)
        for k in range(num_samples):
            n = (
                random.randint(2, len(passages) - 1)
                if include_original
                else random.randint(2, len(passages))
            )
            l = [i + 1 for i in range(len(passages))]
            selected_indexes = random.sample(l, k=n)
            selected_indexes.sort()
            for index in selected_indexes:
                rank_frequencies[index] += 1
            num_passage_stats[n] += 1
            # print("\n\nselected_indexes:\t", selected_indexes)
            prompt = create_prompt(passages, selected_indexes, query)
            # print("new prompt:\t", prompt)

            # Generate new response
            filtered_rankings = {}
            for index in selected_indexes:
                filtered_rankings[index] = rankings[index]
            filtered_rankings = sorted(filtered_rankings.items(), key=lambda x: x[1])
            # print(filtered_rankings)
            new_response = []
            old_new_id_mappings = {}
            for i, old_id in enumerate(selected_indexes):
                old_new_id_mappings[old_id] = i + 1
            for old_id, _ in filtered_rankings:
                new_response.append("[" + str(old_new_id_mappings[old_id]) + "]")
            new_response = " > ".join(new_response)
            # print("new response:\t", new_response)
            new_obj = {}
            new_obj["id"] = obj["id"] + f"_{k}_{n}"
            convs = []
            if len(obj["conversations"]) == 3:
                convs.append(
                    {
                        "from": "system",
                        "value": "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
                    }
                )
            convs.append({"from": "human", "value": prompt})
            convs.append({"from": "gpt", "value": new_response})
            new_obj["conversations"] = convs
            new_objs.append(new_obj)
            num_selected_passages.append(n)
            # print(new_obj)

    print(sum(num_selected_passages) / float(len(num_selected_passages)))
    print(rank_frequencies)
    print(num_passage_stats)
    print(skipped)
    print(len(new_objs))
    # Shuffle new_objs
    random.shuffle(new_objs)
    return new_objs


def write_samples(samples, output_file):
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(samples, json_file)


def main(args):
    data_path = args.data_path
    num_samples = args.num_samples
    num_passages = args.num_passages
    output_file = args.output_file
    include_original = args.include_original
    json_objects = read_data(data_path)
    samples = sample_data(json_objects, num_passages, num_samples, include_original)
    write_samples(samples, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="number of samples generated for each input conversation",
    )
    parser.add_argument(
        "--num_passages",
        type=int,
        default=20,
        help="the number of passages in each prompt of the dataset",
    )
    parser.add_argument(
        "--include_original",
        action="store_true",
        help="whether to include the original prompt in the output",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output file"
    )
    args = parser.parse_args()
    main(args)

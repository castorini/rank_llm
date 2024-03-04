"""Converts and store AWQ-quantized model."""

import argparse
import json
import logging

import awq
import transformers

QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="msp_open_ai_ada2_random_s5000_gpt4_da0_mr20_sampled_mix.jsonl",
        help="Path to the calibration dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="castorini/rank_zephyr_7b_v1_full",
        help="Path/slug to the original model.",
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        default="awq_rank_zephyr_7b_v1_full",
        help="Path/slug where the quantized model is to be stored.")
    args = parser.parse_args()
    return args


def load_dataset(dataset: str):
    """Returns list of prompts for given dataset."""
    with open(dataset, "r") as file:
        data = json.load(file)
    prompts = []
    for content in data:
        content = content["conversations"]
        prompt = ""
        for prompt_dict in content:
            if prompt_dict["from"] == "system":
                prompt += prompt_dict["value"] + "\n"
        for prompt_dict in content:
            if prompt_dict["from"] == "human":
                prompt += prompt_dict["value"] + "\n"
        for prompt_dict in content:
            if prompt_dict["from"] == "gpt":
                prompt += prompt_dict["value"]
        prompts.append(prompt)
    return prompts


def main():
    """Entry point of the script."""
    args = parse_args()
    model_path = args.model_path
    quant_path = args.quant_path
    dataset = args.dataset

    # Load model
    logging.info(f"Loading model from {model_path}.")
    model = awq.AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    logging.info(f"Starting AWQ with data {dataset}.")
    model.quantize(
        tokenizer=tokenizer,
        quant_config=QUANT_CONFIG,
        calib_data=load_dataset(dataset=dataset),
    )

    # Convert config into appropriate format.
    quantization_config = transformers.AwqConfig(
        bits=QUANT_CONFIG["w_bit"],
        group_size=QUANT_CONFIG["q_group_size"],
        zero_point=QUANT_CONFIG["zero_point"],
        version=QUANT_CONFIG["version"].lower(),
    ).to_dict()
    model.model.config.quantization_config = quantization_config

    logging.info(f"Saving quantized model at {quant_path}.")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

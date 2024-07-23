#!/bin/bash

# List of datasets
datasets=("beir_cqadupstack-gis" "beir_cqadupstack-mathematica" "beir_cqadupstack-physics" "beir_cqadupstack-programmers" "beir_cqadupstack-stats" "beir_cqadupstack-tex" "beir_cqadupstack-unix" "beir_cqadupstack-webmasters" "beir_cqadupstack-wordpress" "beir_dbpedia-entity" "beir_fever" "beir_fiqa" "beir_hotpotqa" "beir_nfcorpus" "beir_nq" "beir_quora" "beir_robust04" "beir_scidocs" "beir_scifact" "beir_signal1m" "beir_trec-covid" "beir_trec-news" "beir_webis-touche2020")

# List of GPUs
GPUS=(0 1)

# Function to run the command on a specific GPU
run_on_gpu() {
    dataset=$1
    gpu=$2
    echo "Running dataset: $dataset on GPU: $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python src/rank_llm/scripts/run_rank_llm.py --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=$dataset --batched --retrieval_method=bm25 --prompt_mode=rank_GPT --context_size=4096 --variable_passages
    rm -f "/tmp/gpu_$gpu.lock"
    echo "GPU $gpu freed"
}

# Main loop to iterate over datasets and manage GPU usage
for dataset in "${datasets[@]}"; do
    while true; do
        for gpu in "${GPUS[@]}"; do
            if [ ! -f "/tmp/gpu_$gpu.lock" ]; then
                touch "/tmp/gpu_$gpu.lock"
                run_on_gpu $dataset $gpu &
                break 2
            fi
        done
        # Wait for a few seconds before checking again
        sleep 5
    done
done

# Wait for all background jobs to finish
wait

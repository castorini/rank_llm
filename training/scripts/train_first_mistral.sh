#!/bin/bash

# Define model, dataset paths, and output directory
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TRAIN_DATA_PATH="rryisthebest/rank_zephyr_training_data_alpha"  # Train Dataset --> Hugging Face dataset or Local dataset
OUTPUT_DIR="models/ranking/FirstMistral"  # Directory to save the trained model

mkdir -p "${OUTPUT_DIR}"

DS_SKIP_CUDA_CHECK=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 accelerate launch \
    --config_file "configs/accel_config_deepspeed.yaml" \
    train_rankllm.py \
    --model_name_or_path "${BASE_MODEL}" \
    --train_dataset_path "${TRAIN_DATA_PATH}" \
    --num_train_epochs 3 \
    --seed 42 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 50 \
    --gradient_checkpointing \
    --output_dir "${OUTPUT_DIR}" \
    --noisy_embedding_alpha 5 \
    --objective combined \
    --ranking_loss ranknet \
    --weighted \
    --with_tracking \
    --report_to wandb \
    --checkpointing_steps epoch
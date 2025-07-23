# Training RankLLM

This directory contains the scripts for training the RankLLM model. In addition to training on traditional language modelling, we also support training on learning-to-rank objectives, or a combination of the two.

## Environment

We recommend using a different environment for training than the one you use for running the inference.

To install the dependencies, run:
```bash
conda create -f rank_llm_training_env.yml
conda activate rank_llm_training
pip install flash-attn --no-build-isolation

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd $CONDA_PREFIX/lib
ln -s libcurand.so.10 libcurand.so
cd -
```

## Training

To fine-tune a model for RankLLM, one can run:
```bash
DS_SKIP_CUDA_CHECK=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch train_rankllm.py \
    --model_name_or_path <path-to-model> \
    --train_dataset_path <path-to-train-dataset> \
    --num_train_epochs <num-epochs> \
    --seed <seed> \
    --per_device_train_batch_size <batch-size> \
    --gradient_accumulation_steps <gradient-accumulation-steps> \
    --num_warmup_steps <num-warmup-steps> \
    --gradient_checkpointing \
    --output_dir <output-dir> \
    --noisy_embedding_alpha <noisy-embedding-alpha> \
    --objective <objective>
```
where:
- `<path-to-model>` is the path to the model to fine-tune.
- `<path-to-train-dataset>` is the path to the train dataset
- `<objective>` is one of `generation`, `ranking`, or `combined`. `generation` refers to the traditional language modelling objective, `ranking` refers to the learning-to-rank objective, and `combined` refers to a combination of the two.

## Reproducing Reseults

We have two preset scripts for training [RankZephyr](https://arxiv.org/abs/2312.02724) and [FirstMistral](https://arxiv.org/abs/2411.05508). One can reproduce the results by running:
```bash
bash scripts/train_rank_zephyr.sh
```
for RankZephyr and
```bash
bash scripts/train_first_mistral.sh
```
for FirstMistral.

Once trained, one can run inference with the trained model by running the commands below **from the project root**:
```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=training/models/ranking/RankZephyr/epoch_2 --top_k_candidates=100 --dataset=dl20 \
--retrieval_method=SPLADE++_EnsembleDistil_ONNX --prompt_template_path=src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml --context_size=4096 --variable_passages --num_gpus 1
```
for RankZephyr and
```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=training/models/ranking/FirstMistral/epoch_2 --top_k_candidates=100 --dataset=dl20 --retrieval_method=SPLADE++_EnsembleDistil_ONNX --prompt_template_path=src/rank_llm/rerank/prompt_templates/rank_zephyr_alpha_template.yaml --context_size=4096 --variable_passages --use_logits --use_alpha --num_gpus 1
```
for FirstMistral.




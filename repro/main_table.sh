# Pairs of MODEL AND CHECKPOINT
export MODEL_CHECKPOINT_PATHS=(
    "/u3/rpradeep/local_llms/rank_zephyr_7b_v1_full","/u3/rpradeep/local_llms/rank_zephyr_7b_v1_full"
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-7B-v0.2/checkpoint-324"
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-sampled-mix-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-sampled-mix-7B-v0.2/checkpoint-223"
    )
# "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-and-disc-s2000-sampled-mix-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-and-disc-s2000-sampled-mix-7B-v0.2/checkpoint-89"
# "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-and-disc-s2000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-and-disc-s2000-7B-v0.2/checkpoint-64"

# Enter choice by commandline
CHOICE=${1:-0}
# Variable Passage flag is set to true if CHOICE is 0 or 2
if [ $CHOICE -eq 0 ] || [ $CHOICE -eq 2 ]; then
    VARIABLE_PASSAGE=true
else
    VARIABLE_PASSAGE=false
fi
# Assign MODEL and CHECKPOINT
IFS=',' read -r -a array <<< "${MODEL_CHECKPOINT_PATHS[$CHOICE]}"
MODEL=${array[0]}
CHECKPOINT=${array[1]}


cd ~/rank_llm_private

# Define an array of GPUs
GPUS=(0 1 2 3 4 5)

i=0
# Loop through the sets and topk values
for SET in dl19 dl20 dl21 dl22 news covid; do
    for topk in 100; do
        for FSTAGE in SPLADE++_EnsembleDistil_ONNX; do
            # Loop through the checkpoints and GPUs simultaneously
                checkpoint=${CHECKPOINT}
                gpu=${GPUS[$i]}
                echo "Processing checkpoint: $checkpoint on GPU: $gpu"
                echo $i
                # Set the GPU to be used
                cvd $gpu
                i=$((i+1))

                # Copy necessary files
                cp ${MODEL}/config.json ${MODEL}/added_tokens.json ${MODEL}/special_tokens_map.json ${MODEL}/tokenizer_config.json ${MODEL}/tokenizer.model ${checkpoint}/

                # Modify config.json and run the task in the background
                sed -i 's/"use_cache": false/"use_cache": true/' ${MODEL}/config.json
                echo ${SET} ${topk} ${i}
                if [ $VARIABLE_PASSAGE = true ]; then
                    echo "Variable Passage"
                    python3 rank_llm/run_rank_llm.py --model_path $checkpoint --dataset ${SET} --prompt_mode rank_GPT --retrieval_method ${FSTAGE} --top_k_candidates ${topk} --variable_passage --num_passes 3 &
                else
                    python3 rank_llm/run_rank_llm.py --model_path $checkpoint --dataset ${SET} --prompt_mode rank_GPT --retrieval_method ${FSTAGE} --top_k_candidates ${topk} --num_passes 3 &
                fi
        done
    done
done

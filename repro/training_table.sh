# Pairs of MODEL AND CHECKPOINT
export MODEL_CHECKPOINT_PAITS=(
    "/u3/rpradeep/axolotl/RankZephyrB-7B-v0.1-out-v2","/u3/rpradeep/axolotl/RankZephyrB-7B-v0.1-out-v2/checkpoint-4941"
    "/u3/rpradeep/axolotl/RankZephyrB-bm25-random-s1000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-bm25-random-s1000-7B-v0.2/checkpoint-68"
    "/u3/rpradeep/axolotl/RankZephyrB-bm25-discriminative-s1000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-bm25-discriminative-s1000-7B-v0.2/checkpoint-68"
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s1000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s1000-7B-v0.2/checkpoint-60"
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-discriminative-s1000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-discriminative-s1000-7B-v0.2/checkpoint-56"
    )

# Enter choice by commandline
CHOICE=${1:-0}
# Variable Passage flag is set to true if CHOICE is 0

# Assign MODEL and CHECKPOINT
IFS=',' read -r -a array <<< "${MODEL_CHECKPOINT_PAITS[$CHOICE]}"
MODEL=${array[0]}
CHECKPOINT=${array[1]}


cd ~/rank_llm_private

# Define an array of GPUs
GPUS=(0 1 2 3)

i=0
# Loop through the sets and topk values
for topk in 100; do
    for SET in dl19 dl20; do
        for FSTAGE in openai-ada2 bm25; do
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

                    python3 rank_llm/run_rank_llm.py --model_path $checkpoint --dataset ${SET} --prompt_mode rank_GPT --retrieval_method ${FSTAGE} --top_k_candidates ${topk} &
                    # If i is 4, wait for all the processes to finish
                    if [ $i -eq 4 ]; then
                        wait
                        i=0
                    fi
        done
    done
done


# Check if second argument exists and is eval if so run these else ignore

# Path: repro/training_table.sh

if [ -z "$2" ]
  then
    echo "No second argument supplied"
    exit 1
fi

export MODEL_CHECKPOINT_PAITS=(
    "/u3/rpradeep/axolotl/RankZephyrB-7B-v0.1-out-v2","/u3/rpradeep/axolotl/RankZephyrB-7B-v0.1-out-v2/checkpoint-4941"
    "/u3/rpradeep/axolotl/RankZephyrB-bm25-random-s1000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-bm25-random-s1000-7B-v0.2/checkpoint-68"
    "/u3/rpradeep/axolotl/RankZephyrB-bm25-discriminative-s1000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-bm25-discriminative-s1000-7B-v0.2/checkpoint-68"
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s1000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s1000-7B-v0.2/checkpoint-60"
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-discriminative-s1000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-discriminative-s1000-7B-v0.2/checkpoint-56"
    )

# READ this in for loop

for i in {0..4}; do
    IFS=',' read -r -a array <<< "${MODEL_CHECKPOINT_PAITS[$i]}" 
    MODEL=${array[0]}
    CHECKPOINT=${array[1]}
    MODEL_BASENAME=$(basename $MODEL)
    CHECKPOINT_BASENAME=$(basename $CHECKPOINT)
    for FSTAGE in BM25 OPEN_AI_ADA2; do
    for file in rerank_results/${FSTAGE}/${MODEL_BASENAME}_${CHECKPOINT_BASENAME}*window*; do
        echo $file;
        # dl19 in filename do $TEVAL_MSP_DL19 else $TEVAL_MSP_DL20
        if [[ $file == *"dl19"* ]]; then
            echo "dl19"
            TEVAL_EX=$TEVAL_MSP_DL19
        else
            echo "dl20"
            TEVAL_EX=$TEVAL_MSP_DL20
        fi
        ${TEVAL_EX} $file | egrep 'ndcg_cut_10\s';
    done
    done
done
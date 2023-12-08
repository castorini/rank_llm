# Pairs of MODEL AND CHECKPOINT
export MODEL_CHECKPOINT_PAITS=(
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-sampled-mix-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-sampled-mix-7B-v0.2/checkpoint-223"
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-7B-v0.2/checkpoint-324"
)

# Enter choice by commandline
CHOICE=${1:-0}
# Variable Passage flag is set to true if CHOICE is 0
if [ $CHOICE -eq 0 ]; then
    VARIABLE_PASSAGE=true
else
    VARIABLE_PASSAGE=false
fi
# Assign MODEL and CHECKPOINT
IFS=',' read -r -a array <<< "${MODEL_CHECKPOINT_PAITS[$CHOICE]}"
MODEL=${array[0]}
CHECKPOINT=${array[1]}


cd ~/rank_llm_private

# Define an array of GPUs
GPUS=(0 1 2 3 4 5 6 7)

i=0
# Loop through the sets and topk values
for topk in 20; do
    for SET in dl19 dl20; do
        for FSTAGE in bm25 rep-llama openai-ada2 SPLADE++_EnsembleDistil_ONNX; do
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
                    python3 rank_llm/run_rank_llm.py --model_path $checkpoint --dataset ${SET} --prompt_mode rank_GPT --retrieval_method ${FSTAGE} --top_k_candidates ${topk} --variable_passage &
                else
                    python3 rank_llm/run_rank_llm.py --model_path $checkpoint --dataset ${SET} --prompt_mode rank_GPT --retrieval_method ${FSTAGE} --top_k_candidates ${topk} &
                fi
        done
    done
done

# Pairs of MODEL AND CHECKPOINT
export MODEL_CHECKPOINT_PAITS=(
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-sampled-mix-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-sampled-mix-7B-v0.2/checkpoint-223"
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-s5000-7B-v0.2/checkpoint-324"
)
for i in {0..0}; do
    IFS=',' read -r -a array <<< "${MODEL_CHECKPOINT_PAITS[$i]}" 
    MODEL=${array[0]}
    CHECKPOINT=${array[1]}
    MODEL_BASENAME=$(basename $MODEL)
    CHECKPOINT_BASENAME=$(basename $CHECKPOINT)
    for FSTAGE in BM25 SPLADE_P_P_ENSEMBLE_DISTIL REP_LLAMA OPEN_AI_ADA2; do
    for file in rerank_results/${FSTAGE}/${MODEL_BASENAME}_${CHECKPOINT_BASENAME}*_rank*window_20*; do
        # dl19 in filename do $TEVAL_MSP_DL19 else $TEVAL_MSP_DL20
        if [[ $file == *"shuffled"* ]]; then
            continue
        fi
        if [[ $file == *"pass_"* ]]; then
            continue
        fi
        if [[ $file == *"dl19"* ]]; then
            echo "dl19"
            TEVAL_EX=$TEVAL_MSP_DL19
        elif [[ $file == *"dl20"* ]]; then
            echo "dl20"
            TEVAL_EX=$TEVAL_MSP_DL20
        else
            continue
        fi
    
        echo $file;

        ${TEVAL_EX} $file | egrep 'ndcg_cut_10\s|map_cut_100\s';
    done
    done
done

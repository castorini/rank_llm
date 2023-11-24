# Pairs of MODEL AND CHECKPOINT
export MODEL_CHECKPOINT_PAITS=(
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-and-disc-s2000-sampled-mix-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-and-disc-s2000-sampled-mix-7B-v0.2/checkpoint-89"
    "/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-and-disc-s2000-7B-v0.2","/u3/rpradeep/axolotl/RankZephyrB-open_ai_ada2-random-and-disc-s2000-7B-v0.2/checkpoint-64")

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
GPUS=(0 1 2)

i=0
# Loop through the sets and topk values
for SET in dl19 dl20; do
    for topk in 100; do
        for FSTAGE in bm25 rep-llama SPLADE++_EnsembleDistil_ONNX; do
            i=0
            # Loop through the checkpoints and GPUs simultaneously
            # Loop through window size and step size together like zip over (20, 10), (10, 5), (2, 1)
            list_pairs=(20,10 10,5 2,1)
            for pair in ${list_pairs[@]}; do
                IFS=',' read -r -a array <<< "$pair"
                window_size=${array[0]}
                step_size=${array[1]}
                checkpoint=${CHECKPOINT}
                gpu=${GPUS[$i]}
                echo "Processing checkpoint: $checkpoint on GPU: $gpu"
                echo $i
                i=$((i+1))
                # Set the GPU to be used
                cvd $gpu

                # Copy necessary files
                cp ${MODEL}/config.json ${MODEL}/added_tokens.json ${MODEL}/special_tokens_map.json ${MODEL}/tokenizer_config.json ${MODEL}/tokenizer.model ${checkpoint}/

                # Modify config.json and run the task in the background
                sed -i 's/"use_cache": false/"use_cache": true/' ${MODEL}/config.json
                echo ${SET} ${topk} ${i}
                if [ $VARIABLE_PASSAGE = true ]; then
                    python3 rank_llm/run_rank_llm.py --model_path $checkpoint --dataset ${SET} --prompt_mode rank_GPT --retrieval_method ${FSTAGE} --top_k_candidates ${topk} --variable_passages  \
                    --window_size ${window_size} --step_size ${step_size} &
                else
                    python3 rank_llm/run_rank_llm.py --model_path $checkpoint --dataset ${SET} --prompt_mode rank_GPT --retrieval_method ${FSTAGE} --top_k_candidates ${topk} \
                    --window_size ${window_size} --step_size ${step_size} &
                fi
                if [ $i -eq 3 ]; then
                    wait
                    i=0
                fi
            done
        done
    done
done


for file in rerank_results/*/*dl19*window*; do echo $file; $TEVAL_MSP_DL19 $file | egrep 'ndcg_cut_10|map_cut_100\s'; done;
for file in rerank_results/*/*dl20*window*; do echo $file; $TEVAL_MSP_DL20 $file | egrep 'ndcg_cut_10|map_cut_100\s'; done;

for file in rerank_results/*/*dl19*window*; do echo $file; $TEVAL_MSP_DL19 $file | egrep 'ndcg_cut_10'; done;
for file in rerank_results/*/*dl20*window*; do echo $file; $TEVAL_MSP_DL20 $file | egrep 'ndcg_cut_10'; done;
#!/bin/bash

# Define test cases:
# Format:
#   TEST_NAMES=("Test 1" "Test 2" ...)
#   TEST_COMMANDS=("command1" "command2" ...)
#   TEST_EXPECTED_SCORES=(0.123 0.456 ...)

TEST_NAMES=(
  "FirstMistral (Alpha, Logits)"
  "RZ"
  "Qwen (Alpha)"
  "Monot5"
  "Duot5"
)

TEST_COMMANDS=(
  "python src/rank_llm/scripts/run_rank_llm.py --model_path=castorini/first_mistral --top_k_candidates=50 --dataset=dl19 --retrieval_method=bm25 --context_size=4096 --use_alpha --use_logits --max_queries=3"
  "python src/rank_llm/scripts/run_rank_llm.py --model_path=castorini/rank_zephyr_7b_v1_full  --top_k_candidates=50 --dataset=dl20  --retrieval_method=SPLADE++_EnsembleDistil_ONNX  --context_size=4096 --max_queries=3"
  "python src/rank_llm/scripts/run_rank_llm.py  --model_path=Qwen/Qwen2.5-7B-Instruct --top_k_candidates=50 --dataset=dl21 --retrieval_method=bm25  --context_size=4096 --variable_passages --max_queries=3"
  "python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/monot5-3b-msmarco-10k --top_k_candidates=50 --dataset=dl22 --retrieval_method=bm25  --context_size=4096 --variable_passages --max_queries=3"
   "python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/duot5-3b-msmarco-10k --top_k_candidates=50 --dataset=dl23 --retrieval_method=bm25  --context_size=4096 --variable_passages --max_queries=1"
)

TEST_EXPECTED_SCORES=(
  0.8085
  0.7662
  0.7157
  0.3997
  0.7246
)

for i in "${!TEST_NAMES[@]}"; do
  NAME="${TEST_NAMES[$i]}"
  COMMAND="${TEST_COMMANDS[$i]}"
  EXPECTED_SCORE="${TEST_EXPECTED_SCORES[$i]}"

  echo "Running $NAME..."

  OUTPUT=$(eval "$COMMAND" 2>&1)

  SCORE=$(echo "$OUTPUT" | grep -oP 'ndcg_cut_10\s+all\s+\K\d+\.\d+')

  if [ -z "$SCORE" ]; then
    echo "❌ ERROR: Could not extract nDCG@10 score for '$NAME'"
    continue
  fi

  LOWER_BOUND=$(echo "$EXPECTED_SCORE * 0.975" | bc -l)
  UPPER_BOUND=$(echo "$EXPECTED_SCORE * 1.025" | bc -l)
  PASSED=$(echo "$SCORE >= $LOWER_BOUND && $SCORE <= $UPPER_BOUND" | bc -l)

  if [ "$PASSED" -eq 1 ]; then
    echo "$NAME: PASS ✅ (Actual Score: $SCORE, Expected Score: $EXPECTED_SCORE)"
  else
    echo "$NAME: FAIL ❌ (Actual Score: $SCORE, Expected Score: $EXPECTED_SCORE)"
  fi
done

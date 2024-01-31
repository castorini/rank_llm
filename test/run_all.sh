echo "Running tests..."
tests=("retrieve/test_PyseriniRetriever.py" "rerank/test_RankListwiseOSLLM.py" "rerank/test_SafeOpenai.py")
for test in "${tests[@]}"
do
    echo "Running $test tests..."
    python3 "$test"
    echo 
done
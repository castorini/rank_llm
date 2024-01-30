echo "Running tests..."
tests=("PyseriniRetriever" "RankListwiseOSLLM" "SafeOpenai")
for test in "${tests[@]}"
do
    echo "Running $test tests..."
    python3 run_"$test"_tests.py
    echo 
done
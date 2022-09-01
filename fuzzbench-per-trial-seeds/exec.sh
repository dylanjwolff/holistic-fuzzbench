PYTHONPATH=. python3 experiment/run_experiment.py \
        --experiment-config example-seed-per-trial-experiment.yaml \
        --benchmarks $(cat oss.txt) \
        --experiment-name app \
        --fuzzers aflplusplus --concurrent-builds 1 --allow-uncommitted-changes

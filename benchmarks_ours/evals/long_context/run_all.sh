#!/bin/bash
all_datasets=("long_context")
all_models=("meta-llama/Meta-Llama-3.1-8B-Instruct")
all_approaches=("kvlink-16")

for dataset in ${all_datasets[@]}; do
    for model in ${all_models[@]}; do
        for approach in ${all_approaches[@]}; do
            command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
            echo "Running command: ${command}"
            ${command}
        done
    done
done


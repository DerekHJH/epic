#!/bin/bash
all_datasets=("2wikimqa" "musique" "samsum" "multi_news" "hotpotqa" "needle")
all_models=("mistralai/Mistral-7B-Instruct-v0.2" "meta-llama/Meta-Llama-3.1-8B-Instruct" "01-ai/Yi-Coder-9B-Chat")
all_approaches=("kvlink-0")

for dataset in ${all_datasets[@]}; do
    for model in ${all_models[@]}; do
        for approach in ${all_approaches[@]}; do
            command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
            echo "Running command: ${command}"
            ${command}
        done
    done
done


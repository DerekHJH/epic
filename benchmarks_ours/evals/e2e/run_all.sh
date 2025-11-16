#!/bin/bash
SessionName="e2e"
all_datasets=("2wikimqa" "musique" "samsum" "multi_news" "hotpotqa" "needle")
# all_models=("mistralai/Mistral-7B-Instruct-v0.2" "meta-llama/Meta-Llama-3.1-8B-Instruct" "01-ai/Yi-Coder-9B-Chat")
all_models=("mistralai/Mistral-7B-Instruct-v0.2")

# Create a new tmux session
tmux new-session -d -s ${SessionName}

# Create len(all_datasets) * len(all_models) windows in tmux
counter=2
for dataset in ${all_datasets[@]}; do
    for model in ${all_models[@]}; do
        # Create a new window for each dataset-model pair
        if [ $counter -eq 0 ]; then
            tmux rename-window -t ${SessionName}:${counter} "${dataset:0:4}-${model:0:4}"
        else
            tmux new-window -t ${SessionName}:${counter} -n "${dataset:0:4}-${model:0:4}"
        fi

        # Set the GPU ID for the new window
        tmux send-keys -t ${SessionName}:${counter} "export CUDA_VISIBLE_DEVICES=${counter}" C-m

        # Run the command in the new window
        tmux send-keys -t ${SessionName}:${counter} "bash run_one.sh ${dataset} ${model}" C-m

        counter=$((counter+1))
    done
done

# Attach to the tmux session
tmux attach-session -t ${SessionName}
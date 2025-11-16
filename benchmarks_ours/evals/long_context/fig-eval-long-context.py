import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Union
import string
import json
from benchmarks_ours.evals.utils import approach_names_map

dataset = "long_context"
model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
all_approaches = ['fr', 'cacheblend-15', 'kvlink-16']

if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    last_model_name = model.split("/")[-1]
    path = f"results/{dataset}/{last_model_name}/data.json"
    dataset = pd.read_json(path)

    for approach in all_approaches:
        plt.plot(dataset[f'context_length'], dataset[f'TTFT_{approach}'], label=approach_names_map[approach], linewidth=2)

    plt.xlabel('Context length (# tokens)', fontsize=14)
    plt.ylabel('TTFT (s)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig("results/fig-eval-long-context.pdf", format='pdf', bbox_inches='tight')
        
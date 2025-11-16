import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import List, Union
import string
import json
from benchmarks_ours.evals.utils import model_names_map, dataset_names_map, dataset_metrics_map, approach_names_map


all_datasets = ["hotpotqa"]
all_models = ["meta-llama/Meta-Llama-3.1-8B-Instruct"] 
all_approaches = ['fr', 'naive', 'cacheblend-20', 'cacheblend-15', 'cacheblend-10', 'cacheblend-5', 'cacheblend-1', 'kvlink-32', 'kvlink-16', 'kvlink-8', 'kvlink-4', 'kvlink-2']


colors = ['#33CC33'] + ['#FF3333'] + ['#CC6600', '#FF8000', '#FF9933', '#FFB366', '#FFCC99'] + ['#0076A8', '#2A9BD5', '#4DA6D6', '#7FB3D5', '#A3C1E0']
markers = ['o'] * 2 +  ['s'] * 5 + ['*'] * 5

max_y_metric = 0
max_x_metric = 0


if __name__ == "__main__":

    fig, axs = plt.subplots(
        len(all_datasets), len(all_models), figsize=(6 * len(all_models), 5 * len(all_datasets))
    )

    for i, dataset_name in enumerate(all_datasets):
        for j, model_name in enumerate(all_models):
            last_model_name = model_name.split("/")[-1]
            path = f"results/{dataset_name}/{last_model_name}/data.json"
            dataset = pd.read_json(path)

            for k, approach_name in enumerate(all_approaches):
                if f"score_{approach_name}" not in dataset.columns:
                    continue
                # Could change the metric here
                x_metric = np.mean(dataset[f"TTFT_{approach_name}"])
                y_metric = np.mean(dataset[f"score_{approach_name}"])
                max_y_metric = max(max_y_metric, y_metric)
                max_x_metric = max(max_x_metric, x_metric)
                axs.scatter(
                    x_metric,
                    y_metric,
                    label=approach_names_map[approach_name],
                    s=100,
                    c=colors[k],
                    marker=markers[k],
                )
    # After max_x_metric and max_y_metric are determined
    max_x_metric *= 1.05
    max_y_metric *= 1.05
    for i, dataset_name in enumerate(all_datasets):
        for j, model_name in enumerate(all_models):
            axs.set_xlim(left=0, right=max_x_metric)
            axs.set_ylim(bottom=0, top=max_y_metric)
            axs.tick_params(axis="x", labelsize=16)
            axs.tick_params(axis="y", labelsize=16)

    # axs.legend(loc="upper center", bbox_to_anchor=(0.4, 1.4), ncol=3, fontsize=12)
    axs.legend(loc="lower right", ncol=2, fontsize=13)

    # Draw model names
    for j, model_name in enumerate(all_models):
        axs.set_title(model_names_map[model_name], fontsize=20)

    # Draw x metric name
    for j in range(1, len(all_models) + 1):
        axs.set_xlabel("TTFT / s", fontsize=20)

    for i, dataset_name in enumerate(all_datasets):
        axs.set_ylabel(dataset_metrics_map[dataset_name], fontsize=20)
        axs.text(
            -0.25,
            0.5,
            dataset_names_map[dataset_name],
            fontsize=20,
            rotation="vertical",
            ha="left",
            va="center",
            transform=axs.transAxes,
        )

    # Save
    plt.savefig("results/fig-intro-positioning.png", format="png", bbox_inches="tight", dpi=300)
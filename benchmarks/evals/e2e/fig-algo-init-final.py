import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import List, Union
import string
import json
from benchmarks_ours.evals.utils import model_names_map, dataset_names_map, dataset_metrics_map


all_datasets = ['2wikimqa', 'musique', 'samsum', 'multi_news', 'hotpotqa', 'needle'][::-1]
all_models = ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B-Instruct", "01-ai/Yi-Coder-9B-Chat"] 
all_approaches = ['kvlink-32', 'kvlink-16', 'kvlink-8', 'kvlink-4', 'kvlink-2']


colors =  ['#CC6600', '#FF8000', '#FF9933', '#FFB366', '#FFCC99'] + ["#CC0000", "#FF3333", "#FF6666", "#FF9999", "#FFCCCC"] + ['#0076A8', '#2A9BD5', '#4DA6D6', '#7FB3D5', '#A3C1E0']
markers =  ['s'] * 5 + ['o'] * 5 + ['*'] * 5


if __name__ == "__main__":

    fig, axs = plt.subplots(
        len(all_datasets), len(all_models), figsize=(6 * len(all_models), 5 * len(all_datasets))
    )

    for i, dataset_name in enumerate(all_datasets):
        max_y_metric = 0
        max_x_metric = 0
        for j, model_name in enumerate(all_models):
            last_model_name = model_name.split("/")[-1]
            for u, root_path, in enumerate(["results_init", "results_final", "results_e2e"]):
                path = f"{root_path}/{dataset_name}/{last_model_name}/data.json"
                dataset = pd.read_json(path)

                for k, approach_name in enumerate(all_approaches):
                    if f"score_{approach_name}" not in dataset.columns:
                        continue

                    if "init" in root_path:
                        approach_name = approach_name.split("-")[0] + "-" + str(int(approach_name.split("-")[1]) * 2)
                    # Could change the metric here
                    x_metric = np.mean(dataset[f"TTFT_{approach_name}"])
                    y_metric = np.mean(dataset[f"score_{approach_name}"])
                    max_y_metric = max(max_y_metric, y_metric)
                    max_x_metric = max(max_x_metric, x_metric)
                    axs[i][j].scatter(
                        x_metric,
                        y_metric,
                        label=approach_name if "e2e" in root_path else approach_name + "-" + root_path.split("_")[-1],
                        s=200,
                        c=colors[u * 5 + k],
                        marker=markers[u * 5 + k],
                    )
        # After max_x_metric and max_y_metric are determined
        max_x_metric *= 1.05
        max_y_metric *= 1.05
        for j, model_name in enumerate(all_models):
            axs[i][j].set_xlim(left=0, right=max_x_metric)
            axs[i][j].set_ylim(bottom=0, top=max_y_metric)
            axs[i][j].tick_params(axis="x", labelsize=20)
            axs[i][j].tick_params(axis="y", labelsize=20)

    axs[0][0].legend(loc="upper center", bbox_to_anchor=(1.5, 1.6), ncol=5, fontsize=22)

    # Draw model names
    for j, model_name in enumerate(all_models):
        axs[0][j].set_title(model_names_map[model_name], fontsize=24)

    # Draw x metric name
    for j in range(1, len(all_models) + 1):
        axs[-1][-j].set_xlabel("TTFT / s", fontsize=24)

    for i, dataset_name in enumerate(all_datasets):
        axs[i][0].set_ylabel(dataset_metrics_map[dataset_name], fontsize=24)
        axs[i][0].text(
            -0.35,
            0.5,
            dataset_names_map[dataset_name],
            fontsize=24,
            rotation="vertical",
            ha="left",
            va="center",
            transform=axs[i][0].transAxes,
        )

    # Save
    plt.savefig("results_e2e/fig-algo-init-final.pdf", format="pdf", bbox_inches="tight", dpi=300)
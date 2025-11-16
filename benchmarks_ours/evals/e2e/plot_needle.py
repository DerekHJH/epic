import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import os
from benchmarks_ours.test_type.e2e.configs import Configs
from transformers import AutoTokenizer
from benchmarks_ours.data_sets.utils import str2class

def plot_figure():
    """
    Plot individual figures
    """
    for dataset in ['needle']:
        for model in Configs.all_models:
            for approach in Configs.all_approaches:
                configs = Configs(dataset, model, approach)
                # Yi model does not add bos token by default, so we add it here
                tokenizer = AutoTokenizer.from_pretrained(configs.model, add_bos_token=True)
                df = str2class[configs.dataset](tokenizer=tokenizer, path=configs.result_folder).data

                df = df[['context_length', 'depth_percent', f'score_{approach}']]

                # Create pivot table
                pivot_table = pd.pivot_table(df, values=f'score_{approach}', index=['depth_percent', 'context_length'], aggfunc='mean').reset_index() # This will aggregate
                pivot_table = pivot_table.pivot(index="depth_percent", columns="context_length", values=f'score_{approach}') # This will turn into a proper pivot
                
                
                # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
                # cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

                # Create the heatmap with better aesthetics
                plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
                sns.heatmap(
                    pivot_table,
                    # annot=True,
                    fmt="g",
                    cmap='viridis',  # Continuous colormap
                    cbar_kws={'label': f'score_{approach}'},
                    vmin=0,  # Minimum value for the colormap
                    vmax=1,  # Maximum value for the colormap
                )

                # More aesthetics
                plt.title(f'{configs.dataset} {configs.model} {configs.approach}')  # Adds a title
                plt.xlabel('Context length')  # X-axis label
                plt.ylabel('Depth percent')  # Y-axis label
                plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
                plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
                plt.tight_layout()  # Fits everything neatly into the figure area

                # Save the plot
                plt.savefig(os.path.join(configs.result_folder, f'{approach}.png'))

def plot_figures():
    """
    Plot all figures ont the same figure
    """
    fig, axes = plt.subplots(len(Configs.all_approaches), len(Configs.all_models), \
                            figsize=(6 * len(Configs.all_models), 3.5 * len(Configs.all_approaches)))

    for dataset in ['needle']:
        for i, approach in enumerate(Configs.all_approaches):
            for j, model in enumerate(Configs.all_models):
                ax = axes[i][j]
                configs = Configs(dataset, model, approach)
                # Yi model does not add bos token by default, so we add it here
                tokenizer = AutoTokenizer.from_pretrained(configs.model, add_bos_token=True)
                df = str2class[configs.dataset](tokenizer=tokenizer, path=configs.result_folder).data

                df = df[['context_length', 'depth_percent', f'score_{approach}']]

                # Create pivot table
                pivot_table = pd.pivot_table(df, values=f'score_{approach}', index=['depth_percent', 'context_length'], aggfunc='mean').reset_index() # This will aggregate
                pivot_table = pivot_table.pivot(index="depth_percent", columns="context_length", values=f'score_{approach}') # This will turn into a proper pivot

                sns.heatmap(
                    pivot_table,
                    # annot=True,
                    fmt="g",
                    cmap='viridis',  # Continuous colormap,
                    # cbar=(j == len(Configs.all_models) - 1), # Only show colorbar for the last column
                    cbar_kws={'label': f'score_{approach}'},
                    vmin=0,  # Minimum value for the colormap
                    vmax=1,  # Maximum value for the colormap
                    ax=ax
                )

                if i == 0:
                    ax.set_title(model.split('/')[-1])
                
                if i == len(Configs.all_approaches) - 1:
                    ax.set_xlabel('Context length')
                    ax.set_xticklabels(pivot_table.columns.to_list(), rotation=45)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel('')
                    ax.tick_params(axis='x', which='both', length=0)
                if j == 0:
                    ax.set_ylabel('Depth percent')
                    ax.set_yticklabels(pivot_table.index.to_list(), rotation=0)
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
                    ax.tick_params(axis='y', which='both', length=0)


    # Save the plot
    plt.tight_layout()
    plt.savefig(f'results/needle.pdf', format="pdf", bbox_inches='tight')

if __name__ == "__main__":
    plot_figures()



import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import List
import os
from matplotlib.colors import TwoSlopeNorm

dataset = "dummy"
model = "meta-llama/Meta-Llama-3.1-8B-Instruct".split("/")[-1]
approach_name = "kvlink-0"
root_path = f"results/{dataset}/{model}"

layer_id = 5
head_id = 5

def squre_attnmap(attnmap: List[List[torch.Tensor]]):
    
    assert len(attnmap) == 32, "The number of layers are not 32"
    assert len(attnmap[0]) <= 10, "The sequence length is longer than 10"
    assert len(attnmap[0][0].shape) == 4, "The shape is not (num_kv_head_group, num_kv_head_per_group, 1, seq_len)"
    assert attnmap[0][0].shape[-1] + 1 == attnmap[0][1].shape[-1], "The sequence is not increasing"
    # Move to CPU and convert bfloat to float
    for i in range(len(attnmap)):
        for j in range(len(attnmap[i])):
            # Min-max scale the attention score instead of using softmax to make the visualization more clear
            # attnmap = torch.softmax(attnmap, dim=-1)
            attnmap[i][j] = (attnmap[i][j] - attnmap[i][j].min(-1, keepdim=True).values) / (attnmap[i][j].max(-1, keepdim=True).values - attnmap[i][j].min(-1, keepdim=True).values)

            padding = (0, attnmap[0][-1].shape[-1] - attnmap[i][j].shape[-1])
            attnmap[i][j] = torch.nn.functional.pad(attnmap[i][j], padding, mode="constant", value=0)

    attnmap = [torch.cat(m, dim=-2).reshape(attnmap[0][0].shape[0] * attnmap[0][0].shape[1], -1, attnmap[0][0].shape[3]).cpu().float() for m in attnmap]

    return attnmap

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=300)
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    attnmap = torch.load(os.path.join(root_path, f"attn_matrix_{approach_name}.pt"))
    start_offset = torch.load(os.path.join(root_path, f"start_offset_{approach_name}.pt"))
    attnmap = squre_attnmap(attnmap)
    attnmap = attnmap[layer_id][head_id]

    cax = ax.imshow(attnmap, cmap='viridis', norm=norm)

    ax.set_xticks(start_offset)
    ax.set_xticklabels(start_offset, fontsize=15)
    # ax.set_title(f"Layer {layer_id}, Head {head_id}, Approach {approach_name}", fontsize=12)
    ax.set_yticks([])
    
    # Set color bar
    # fig.colorbar(cax, ax=ax, orientation='vertical')

# Add color bar for reference
# fig.colorbar(cax)
plt.tight_layout()
plt.savefig("results/fig-eval-zero.pdf", format="pdf", bbox_inches="tight", dpi=300)
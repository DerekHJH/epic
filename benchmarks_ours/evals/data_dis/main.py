from benchmarks_ours.test_type.data_dis.configs import Configs
from transformers import AutoTokenizer
from benchmarks_ours.data_sets.utils import str2class
import numpy as np
import matplotlib.pyplot as plt
import os
from benchmarks_ours.utils import dataset_names_map

import os
import logging
logging.basicConfig(level=logging.INFO)
class Configs:
    cwd = '/home/jovyan/hjh/KVLink/benchmarks_ours/test_type/data_dist' # Current working directory
    all_datasets = ['2wikimqa', 'musique', 'samsum', 'multi_news', 'hotpotqa', 'needle'] #'passage_retrieval_en'
    all_models = ['mistralai/Mistral-7B-Instruct-v0.2']

    def __init__(self, 
        dataset=all_datasets[0],
        model=all_models[0]
    ):

        self.dataset = dataset
        self.model = model
        self._verify_init_args()
        

        # Create parent folders for results
        self.result_folder = os.path.join('results', dataset, model.split('/')[-1])
        os.makedirs(self.result_folder, exist_ok=True)

    def _verify_init_args(self):
        assert self.dataset in self.all_datasets, f'{self.dataset} not in {self.all_datasets}'
        assert self.model in self.all_models, f'{self.model} not in {self.all_models}'


fig, axs = plt.subplots(1, 2, figsize=(12, 4))


for dataset_name in Configs.all_datasets:
    configs = Configs(
        dataset=dataset_name,
    )
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', add_bos_token=True)
    dataset = str2class[dataset_name](path=configs.result_folder, tokenizer=tokenizer)

    if dataset_name == 'needle':
        data_sorted = np.sort(dataset.data['context_length'])
    else:    
        data_sorted = np.sort(dataset.data['length'])
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    axs[0].plot(data_sorted, cdf, label=dataset_names_map[dataset_name])


    data_sorted = np.sort(dataset.data['output_length'])
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    axs[1].plot(data_sorted, cdf, label=dataset_names_map[dataset_name])

    dataset.save_dataset(configs.result_folder)


for ax in axs:
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend(fontsize=12)
    ax.set_ylabel('CDF', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)


axs[0].set_xlabel('Prompt length (tokens)', fontsize=20)
axs[0].set_title('(a) Prompt length distribution', fontsize=20)

axs[1].set_xlabel('Answer length (tokens)', fontsize=20)
axs[1].set_title('(b) Answer length distribution', fontsize=20)

# Show the plot
plt.savefig('results/data_dis.pdf', format='pdf', bbox_inches='tight')
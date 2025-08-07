import os
import logging
logging.basicConfig(level=logging.INFO)
class Configs:
    cwd = '/home/juhu/hjh/KVLink/benchmarks_ours/test_type/reorder' # Current working directory
    all_datasets = ['2wikimqa-beginning', '2wikimqa-middle', '2wikimqa-end', '2wikimqa-original']
    all_models = ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3.1-8B-Instruct', '01-ai/Yi-Coder-9B-Chat']
    all_approaches = ['normal', 'no-kvlink', 'cacheblend-20', 'cacheblend-15', 'cacheblend-10', 'cacheblend-5', 'cacheblend-1', 'kvlink-20', 'kvlink-15', 'kvlink-10', 'kvlink-5', 'kvlink-1'] # Make sure the first approach is the normal one

    def __init__(self, 
        dataset=all_datasets[0], 
        model=all_models[0], 
        approach=all_approaches[0]
    ):

        self.dataset = dataset
        self.model = model
        self.approach = approach  
        self._verify_init_args()
        

        # Create parent folders for results
        self.result_folder = os.path.join('results', dataset, model.split('/')[-1])
        os.makedirs(self.result_folder, exist_ok=True)

        # Other configs
        self.seed = 42

    def _verify_init_args(self):
        assert self.model in self.all_models, f'{self.model} not in {self.all_models}'
        assert self.dataset in self.all_datasets, f'{self.dataset} not in {self.all_datasets}'
        assert self.approach in self.all_approaches, f'{self.approach} not in {self.all_approaches}'
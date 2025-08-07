import pandas as pd
results1_path_root = "e2e/results" # This folder is mutable
results2_path_root = "e2e/results_init" # This folder is imutable
all_datasets = ['2wikimqa', 'musique', 'samsum', 'multi_news', 'hotpotqa', 'needle'][::-1]
all_models = ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B-Instruct", "01-ai/Yi-Coder-9B-Chat"] 
all_approaches = ['kvlink-32', 'kvlink-16', 'kvlink-8', 'kvlink-4', 'kvlink-2', "kvlink-1"]
if __name__ == "__main__":
    
    for dataset in all_datasets:
        for model in all_models:
            last_model_name = model.split("/")[-1]
            path1 = f"{results1_path_root}/{dataset}/{last_model_name}/data.json"
            path2 = f"{results2_path_root}/{dataset}/{last_model_name}/data.json"
            dataset1 = pd.read_json(path1)
            dataset2 = pd.read_json(path2)
            for approach in all_approaches:
                dataset1[f"score_{approach}"] = dataset2[f"score_{approach}"]
                dataset1[f"TTFT_{approach}"] = dataset2[f"TTFT_{approach}"]
            dataset1.to_json(path1)
    

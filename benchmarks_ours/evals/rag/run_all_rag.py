import subprocess
from benchmarks_ours.test_type.rag.configs import Configs

for dataset in Configs.all_datasets:
    for model in Configs.all_models:
        for approach in Configs.all_approaches:

            command = f'python3 main.py --dataset {dataset} --model {model} --approach {approach}'
            print(f'Running command: {command}')
            process = subprocess.Popen(command, shell=True, cwd=Configs.cwd)
            process.wait()


        command = f'python3 main.py --dataset {dataset} --model {model} --eval'
        print(f'Running command: {command}')
        process = subprocess.Popen(command, shell=True, cwd=Configs.cwd)
        process.wait()

command = 'python3 main.py --plot'
print(f'Running command: {command}')
process = subprocess.Popen(command, shell=True, cwd=Configs.cwd)
process.wait()

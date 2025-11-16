# Benchmarks

This folder contains the benchmarks used in the paper. The benchmarks vary mainly in three dimensions:
- Model
- Dataset
- Test type

## Models

The models used in the benchmarks are: Mistral-7B-Instruct-v0.2, ...

Change the model_name in configs.py to switch between models.

- Mistral-7B: 

## Datasets

The datasets used in the benchmarks are: 2WikiMQA, Musique, MultiNews, ...

- 2WikiMQA:
- Musique:
- MultiNews:

Change the dataset_name in configs.py to switch between datasets.

## Test types

The test types used in the benchmarks are: sync, open-loop, ...

Change the test_type in configs.py to switch between test types.

- sync: A synchronous version of LLM invoking. In this mode, LLM serves one request at a time. This mode is mainly used for algorithm validation instead of system performance evaluation.
- open-loop: 


## Run benchmarks

### Environment setup

```bash
# Add benchmark_ours to PYTHONPATH in .bashrc
export PYTHONPATH="/data0/hujunhao/KVLink/":$PYTHONPATH

# Create conda environment 
conda create -n vllm python==3.11
conda activate vllm

# In KVLink folder
pip install -e . 

# In benchmarks_ours folder
pip install -r requirements.txt

# import vllm to check if everything works fine, if not, see the following steps
```

### Environment setup troubleshooting

```bash
# If you see an error about undefined symbol in some .so file, try the following (after finishing all steps in the previous section):

# 1. In KVLink folder
python setup.py build

# 2. Copy the .so file to the vllm folder
cp KVLink/build/lib.linux-x86_64-cpython-311/vllm/_C.cpython-311-x86_64-linux-gnu.so KVLink/vllm/

# 3. import vllm to check if everything works fine
```

### Run benchmarks

```bash 
# In benchmarks_ours folder, find the test_type folder you want to run, e.g., e2e
cd evals/e2e

# To run a single benchmark
bash test.sh

# To run all benchmarks
cd evals/e2e
bash run_all.sh
```



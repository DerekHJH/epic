dataset="dummy"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
all_approaches=("fr" "naive" "cacheblend-15" "kvlink-1")

# Take the arguments from the command line
if [ $# -eq 0 ]; then
    echo "No arguments provided. Using default values."
elif [ $# -eq 1 ]; then
    dataset=$1
elif [ $# -eq 2 ]; then
    dataset=$1
    model=$2
else
    echo "Too many arguments provided. Usage: $0 [dataset] [model]"
    exit 1
fi


for approach in ${all_approaches[@]}; do
    command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
    echo "Running command: ${command}"
    ${command}
done
dataset="2wikimqa"
model="mistralai/Mistral-7B-Instruct-v0.2"
all_approaches=("kvlink-1-31" "kvlink-2-30" "kvlink-4-28" "kvlink-8-24" "kvlink-16-16" "kvlink-24-8" "kvlink-28-4" "kvlink-30-2" "kvlink-31-1")

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
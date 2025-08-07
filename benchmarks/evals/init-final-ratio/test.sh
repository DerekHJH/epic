dataset="2wikimqa"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
approach="kvlink-16-16"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}
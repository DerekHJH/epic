dataset="dummy"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
approach="kvlink-0"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}
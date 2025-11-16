dataset="multi_news"
model="01-ai/Yi-Coder-9B-Chat"
approach="fr"

command="python3 main.py --dataset ${dataset} --model ${model} --approach ${approach}"
echo "Running command: ${command}"
${command}
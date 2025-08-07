import asyncio
import numpy as np
import os
import configs
import json
from workloads.mmlu import MMLU

results = []

flag = True

async def dummy_post_request_and_get_response(configs, client, data, waiting_time) -> None:
    await asyncio.sleep(waiting_time)
    print(data['_id'])

async def post_request_and_get_response(configs, client, data, waiting_time) -> None:
    await asyncio.sleep(waiting_time)
    global flag
    if flag:
        flag = False
        print(data['user_prompt'])
        print('*' * 100)
        print(data['user_prompt_change_position'])

    result = {'answer': data['answer']}
    try:
        completion = await client.chat.completions.create(
            model = configs.model,
            messages = [
                {"role": "user", "content": data['user_prompt_change_position']}
            ],
            logprobs = True,
            max_tokens = 100
        )
        result['output_1'] = completion.choices[0].message.content
        result['logprob_1'] = {token.token: token.logprob for token in completion.choices[0].logprobs.content}
        completion = await client.chat.completions.create(
            model = configs.model,
            messages = [
                {"role": "user", "content": data['user_prompt']}
            ],
            logprobs = True,
            max_tokens = 100
        )
        result['output_2'] = completion.choices[0].message.content
        result['logprob_2'] = {token.token: token.logprob for token in completion.choices[0].logprobs.content}
    except Exception as e:
        print(e)
        return
    results.append(result)


async def run(configs, client):

    dataset = MMLU()
    coroutines = []
    waiting_time = 0

    cnt = 0
    for data in dataset:
        coroutines.append(asyncio.create_task(post_request_and_get_response(configs, client, data, waiting_time)))
        interval = np.random.exponential(1.0 / configs.request_rate)
        waiting_time = waiting_time + interval
        cnt += 1
        if cnt == 10:
            break
    await asyncio.gather(*coroutines)
    
    if not os.path.exists(os.path.dirname(configs.result_path)):
        os.makedirs(os.path.dirname(configs.result_path))
    with open(configs.result_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
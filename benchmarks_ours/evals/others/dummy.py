import asyncio
import numpy as np
import configs
from transformers import AutoTokenizer


async def run(configs, client):

    tokenizer = AutoTokenizer.from_pretrained(configs.model)
    prefix = 'Please help answer one multiple-choice question. There are five examples for your reference. Give me your choice for the unanswered questions starting with Answer:'
    data = '''
        You are using the default legacy behaviour of the 
        <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. 
        This is expected, and simply means that the `legacy` (previous) behavior will be used so 
        nothing changes for you. If you want to use the new behaviour, set `legacy=False`. 
        This should only be set if you understand what it means, and thoroughly read the reason
        why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if 
        you loaded a llama tokenizer from a GGUF file you can ignore this message.
    '''

    haha = tokenizer.encode(prefix, return_tensors="pt")
    import pdb
    pdb.set_trace()

    completion = await client.chat.completions.create(
        model = configs.model,
        messages = [
            {"role": "user", "content": data}
        ],
        max_tokens = 1024,
    )


    completion = await client.chat.completions.create(
        model = configs.model,
        messages = [
            {"role": "user", "content": "" + prefix + data}
        ],
        max_tokens = 1024,
    )
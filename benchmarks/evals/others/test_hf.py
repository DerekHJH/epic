import torch
from typing import List
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        # trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        # torch_dtype=torch.float16,
        # trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer

def run(configs):
    model, tokenizer = load(configs.model)
    # if configs.approach == 'kvlink':
    #     k_seq_dim, v_seq_dim = enable_kvlink(model)

    # Start testing --- We only consider one test case
    system_prompts = 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n'
    mod_prompts = [
        'Derek is a single man living in Qingpu District. Qingpu is beautiful and suitable for wilderness adventures.\n', 
        'All people living in Qingpu District work in Huawei, a great company that cares a lot about its employees and has a very humanistic atmosphere.\n'
    ]
    free_form_prompt = 'Where does Derek work?\n'
    # Convert modules in the prompt to tokens and process them
    # tokenizer.encode(...)[1:] to remove the initial token
    system_token_ids: List[int] = tokenizer.encode(system_prompts)
    mod_token_ids: List[List[int]] = [tokenizer.encode(mod_prompt) for mod_prompt in mod_prompts]
    free_form_token_ids: List[int] = tokenizer.encode(free_form_prompt)
    token_ids: List[List[int]] = [system_token_ids] + mod_token_ids + [free_form_token_ids]


    if configs.approach == 'normal':
        past_key_values = None
        input_ids = [token_id for token_list in token_ids for token_id in token_list] # Compress token_ids into one list
        input_ids = torch.tensor(input_ids, device='cuda').unsqueeze(0)

        start_time = time.perf_counter()
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        end_time = time.perf_counter()
        past_key_values = outputs.past_key_values
    elif configs.approach == 'kvlink':
        past_key_values = None
        all_past_key_values = []
        for i in range(len(token_ids) - 1): # Ignore the last free_form_token_ids
            input_ids = torch.tensor(token_ids[i], device='cuda').unsqueeze(0)
            outputs = model(input_ids)
            all_past_key_values.append(outputs.past_key_values)

        past_key_values = [[all_past_key_values[0][i][j] for j in range(len(all_past_key_values[0][0]))] for i in range(len(all_past_key_values[0]))]
        
        for i in range(len(all_past_key_values[0])):
            for j in range(len(all_past_key_values[0][0])):
                for k in range(1, len(all_past_key_values)): 
                    past_key_values[i][j] = torch.cat((past_key_values[i][j], all_past_key_values[k][i][j]), dim=2)


        input_ids = torch.tensor(token_ids[-1], device='cuda').unsqueeze(0) # free_form_token_ids
        start_time = time.perf_counter()
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        end_time = time.perf_counter()
        past_key_values = outputs.past_key_values

    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    for _ in range(model.config.max_position_embeddings - past_key_values[0][0].shape[2]):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        
        if pred_token_idx == tokenizer.eos_token_id:
            break

    generated_text = (
        tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        )
        .strip()
        .split(" ")
    )
    print(" ".join(generated_text))
    print(f'kvlink TTFT: {end_time-start_time} s')


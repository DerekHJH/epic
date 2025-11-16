from typing import List
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
from benchmarks_ours.data_sets.data_set import Data_set
from benchmarks_ours.evals.utils import str2class, set_seed
import os
from dataclasses import dataclass, field
import logging
logging.basicConfig(level=logging.INFO)
from collections import defaultdict
import numpy as np
from tqdm.contrib import tenumerate

logger = logging.getLogger(__name__)

@dataclass
class EvalConfigs:
    dataset: str
    model: str
    approach: str
    tot_num_data: int = 200
    all_datasets: List[str] = field(
        default_factory = lambda: ['2wikimqa', 'musique', 'samsum', 'multi_news', 'hotpotqa', 'needle']
    ) # 'mmlu', 'passage_retrieval_en', 
    # '01-ai/Yi-6B-200K', '01-ai/Yi-1.5-9B-Chat-16K', '01-ai/Yi-Coder-9B-Chat', '01-ai/Yi-1.5-9B-32K' all the same 
    # For llama2, we need to squeeze the 2 dim in xformers attention_bias and set gpu_utilization to 0.7
    all_models: List[str] = field(
        default_factory=lambda: [
            'mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3.1-8B-Instruct', '01-ai/Yi-Coder-9B-Chat'] 
        )# , 'xverse/XVERSE-13B-256K', 'mistralai/Mistral-Nemo-Instruct-2407', 'mistralai/Mistral-Small-Instruct-2409', 
    all_approaches: List[str] = field(
        default_factory = lambda: ["kvlink-0"]
    ) # Make sure the first approach is the fr one

    model_config: AutoConfig = field(init=False)

    seed: int = 42
    result_path: str = "results"

    @classmethod
    def get_configs_from_cli_args(cls) -> "EvalConfigs":
        """
        Parse the command line arguments and return the Configs object.
        """
        # Add the arguments to the parser.
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--approach", type=str, required=True)

        # Parse the arguments.
        args = parser.parse_args()
        configs = cls(**vars(args))
        return configs

    def __post_init__(self):
        """
        Verify the init arguments and create the result path.
        """
        self._verify_init_args()
        self.result_path = os.path.join(self.result_path, self.dataset, self.model.split("/")[-1])
        os.makedirs(self.result_path, exist_ok=True)

        self.model_config = AutoConfig.from_pretrained(self.model)

    def _verify_init_args(self):
        assert self.model in self.all_models, f"{self.model} not in {self.all_models}"
        assert self.dataset in self.all_datasets, f"{self.dataset} not in {self.all_datasets}"
        assert self.approach in self.all_approaches, f"{self.approach} not in {self.all_approaches}"

class EvalEngine:
    """
    Evaluate a specific approach on a specific model and a specific dataset.
    """

    def __init__(self, configs: EvalConfigs) -> None:
        self.configs = configs

    def run(self):
        logging.info(
            (
                f"Evaluate \033[32m{self.configs.approach}\033[0m on"
                f" \033[32m{self.configs.model}\033[0m and"
                f" \033[32m{self.configs.dataset}\033[0m"
            )
        )
        logging.info(f"Save the results to \033[32m{self.configs.result_path}\033[0m")

        set_seed(configs.seed)

        # Step 1: Preprocessing, load and modify neccessary components such as
        # tokenizer, dataset, model and pipeline.

        # Yi model does not add bos token by default, so we add it here
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.configs.model, add_bos_token=True)
        self.dataset: Data_set = str2class[self.configs.dataset](
            tokenizer=self.tokenizer,
            path=self.configs.result_path,
            tot_num_data=self.configs.tot_num_data,
        )
        self.dataset.save_dataset(self.configs.result_path) # Checkpoint the dataset
        self.model: AutoModelForCausalLM = LLM(
            model=self.configs.model, 
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
        )
        self.model.set_tokenizer(self.tokenizer)

        # Step 2: Run the inference and record results into the dataset
        self.dataset = self.run_inference(self.configs, self.dataset, self.tokenizer, self.model)

        # Step 3: Generate the presentation
        self.generate_presentation()


    def run_inference(self, configs: EvalConfigs, dataset: Data_set, tokenizer: AutoTokenizer, llm: LLM) -> Data_set:
        """
        Run the inference and record the results into the dataset.
        """

        logger.info("Run the inference. This might take a long time... Good luck")

        results = defaultdict(list)

        # Start testing
        len_dummy_tokens = 3
        dummy_tokens = [tokenizer.bos_token_id] * len_dummy_tokens

        for _, (system_prompts, mod_prompts, free_form_prompt) in tenumerate(dataset, desc="dataset", leave=True):
            # Convert modules in the prompt to tokens and process them
            # Each mod contains tokenizer.bos_token_id
            system_token_ids: List[int] = tokenizer.encode(system_prompts)
            mod_token_ids: List[List[int]] = [dummy_tokens + tokenizer.encode(mod_prompt) for mod_prompt in mod_prompts]
            free_form_token_ids: List[int] = dummy_tokens + tokenizer.encode(free_form_prompt)
            token_ids: List[List[int]] = [system_token_ids] + mod_token_ids + [free_form_token_ids]

            # Flatten token_ids, delete all intermediate tokenizer.bos_token_id, keep one tokenizer.bos_token_id at the beginning
            # input_ids = [tokenizer.bos_token_id] + [_token_ids[i] for _token_ids in token_ids for i in range(1, len(_token_ids))]
            
            input_ids = system_token_ids[:len_dummy_tokens+1] + [_token_ids[i] for _token_ids in token_ids for i in range(len_dummy_tokens+1, len(_token_ids))]

            cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
            
            logger.debug('Collecting old kvs')
            cache_fuse_metadata['collect'] = True
            sampling_params = SamplingParams(temperature=0, max_tokens=1)

            old_kvs = []
            for i in range(len(token_ids)):
                llm.generate(
                    sampling_params=sampling_params,
                    prompt_token_ids=[token_ids[i]],
                    use_tqdm=False, # Prevent excessive output
                )
                
                llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
                for j in range(len(llm_layers)):
                    # Delete all intermidiate tokenizer.bos_token_id, keep one tokenizer.bos_token_id at the beginning
                    if i == 0:
                        temp_k = llm_layers[j].self_attn.hack_kv[0].clone()
                        temp_v = llm_layers[j].self_attn.hack_kv[1].clone()
                    else:
                        temp_k = llm_layers[j].self_attn.hack_kv[0][len_dummy_tokens+1:len(token_ids[i])].clone()
                        temp_v = llm_layers[j].self_attn.hack_kv[1][len_dummy_tokens+1:len(token_ids[i])].clone()  

                    if i == 0:
                        old_kvs.append([temp_k, temp_v])
                    else:
                        old_kvs[j][0] = torch.cat((old_kvs[j][0],temp_k), dim=0)
                        old_kvs[j][1] = torch.cat((old_kvs[j][1],temp_v), dim=0)
                llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = old_kvs

            logger.debug('Second inference for performance evaluation')
            start_offset = [0]
            for i, _token_ids in enumerate(token_ids):
                if i == 0:
                    start_offset.append(start_offset[-1] + len(_token_ids))
                else:
                    start_offset.append(start_offset[-1] + len(_token_ids) - len_dummy_tokens - 1)

            assert start_offset[-1] == len(input_ids), f"start_offset[-1]: {start_offset[-1]}, len(input_ids): {len(input_ids)}"
            if 'kvlink' in configs.approach:
                temp = list(range(start_offset[-2], start_offset[-1]))
                cache_fuse_metadata["kvlink"] = temp
                cache_fuse_metadata["check"] = False
                cache_fuse_metadata['collect'] = False

            sampling_params = SamplingParams(
                temperature=0, 
                # max_tokens=10,
            )
            output = llm.generate(
                sampling_params=sampling_params,
                prompt_token_ids=[input_ids],
                use_tqdm=False, # Prevent excessive output
            )
            results[f'output_{configs.approach}'].append(output[0].outputs[0].text)
            results[f'TTFT_{configs.approach}'].append(output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time)
            # print(f"{configs.approach } generation: {output[0].outputs[0].text}")
            # print(f"{configs.approach} TTFT: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")

        dataset.update(results)
        dataset.save_dataset(configs.result_path)

        return dataset

    def generate_presentation(self):
        """
        Present the results by invoking this function after executing run().
        Separating this function from run() improves efficiency by saving execution time.
        The results are saved in self.configs.result_path.
        """
        # TODO(hjh): Implement the presentation function
        self.dataset.calc_accuracy(self.configs.dataset, self.configs.approach)
        self.dataset.save_dataset(self.configs.result_path)

        accuracy_avg = np.mean(self.dataset.data[f"score_{self.configs.approach}"])
        TTFT_avg = np.mean(self.dataset.data[f"TTFT_{self.configs.approach}"])
        logger.info(f"Average accuracy of {self.configs.approach}: {accuracy_avg:.3f}")
        logger.info(f"Average TTFT of {self.configs.approach}: {TTFT_avg:.2f} s")



if __name__ == "__main__":

    configs = EvalConfigs.get_configs_from_cli_args()
    eval_engine = EvalEngine(configs)
    eval_engine.run()

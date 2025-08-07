from benchmarks_ours.test_type.reorder.configs import Configs
from benchmarks_ours.utils import set_seed
import argparse
from transformers import AutoTokenizer
from vllm import LLM
from benchmarks_ours.data_sets.utils import str2class


def get_args() -> argparse.Namespace:
    """ Get arguments from the command line.
    Please refer to the configs.py file for detailed information.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimqa-original', help="The dataset name in LongBench from huggingface")
    parser.add_argument("--model", type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help="The model name from huggingface")
    parser.add_argument("--approach", type=str, default='normal', help="The approach used")
    parser.add_argument("--eval", action='store_true', help="Evaluate the results")
    parser.add_argument("--plot", action='store_true', help="Plot the results")
    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = get_args()
    configs = Configs(
        dataset=args.dataset,
        model=args.model,
        approach=args.approach,
    )
    # Yi model does not add bos token by default, so we add it here
    tokenizer = AutoTokenizer.from_pretrained(configs.model, add_bos_token=True) 
    dataset = str2class[configs.dataset](
        tokenizer=tokenizer,
        path=configs.result_folder,
        ref_doc_pos=configs.dataset.split('-')[1]
    )
    
    if args.eval:
        from benchmarks_ours.tools.eval import run
        run(configs, dataset)
        
    elif args.plot:
        from benchmarks_ours.tools.plot import run
        run(configs)
    else:
        from benchmarks_ours.tools.test import run
        
        set_seed(configs.seed)
        llm = LLM(
            model=configs.model, 
            gpu_memory_utilization=0.8,
        )
        llm.set_tokenizer(tokenizer)

        run(configs, dataset, llm, tokenizer)


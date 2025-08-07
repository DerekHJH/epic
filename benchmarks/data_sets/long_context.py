import logging
import pandas as pd
import os
import numpy as np
from typing import Dict, List
import glob
from transformers import AutoTokenizer
from tqdm import tqdm
from functools import lru_cache
from benchmarks_ours.data_sets.data_set import Data_set

logger = logging.getLogger(__name__)


class LongContext(Data_set):

    def load_data_from_hf(self) -> pd.DataFrame:
        self.haystack_dir = "PaulGrahamEssays"
        self.context_lengths_min = 1000
        self.context_lengths_max = 50000
        self.context_lengths_num_intervals = 50
        self.context_lengths = np.round(np.linspace(self.context_lengths_min, self.context_lengths_max, num=self.context_lengths_num_intervals, endpoint=True)).astype(int)
        
        data = []
        for context_length in tqdm(self.context_lengths):
            context = self.generate_context(context_length)
            data.append(pd.DataFrame({
                'context': [context], 
                'context_length': [context_length],
            }))
        data = pd.concat(data, ignore_index=True)
        return data
    
    def generate_context(self, context_length):
        # Load up tiktoken so we navigate tokens more easily

        # Get your haystack dir files loaded into a string
        context = self.read_context_files()

        # Truncate the haystack dir essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        return context
    

    def get_context_length_in_tokens(self, context):
        return len(self.tokenizer.encode(context))
    
    # Use this because the max context length (32K) is far less than the total length of all the essays 
    # Each time we call this function will get the same results
    @lru_cache() 
    def read_context_files(self):
        context = ''
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

        for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
            with open(file, 'r') as f:
                context += f.read()
            if self.get_context_length_in_tokens(context) > max_context_length:
                break

        assert self.get_context_length_in_tokens(context) > max_context_length, f'Context length of all essays should be greater than {max_context_length}'
        return context

    def encode_and_trim(self, context, context_length):
        tokens = self.tokenizer.encode(context)[1:]  # Remove the first token (BOS token)
        if len(tokens) > context_length:
            context = self.tokenizer.decode(tokens[:context_length])
        
        return context
    
    def _append_input(self, row: Dict) -> Dict:
        row['input'] = '\nGenerate at most one word:'
        return row
    
    def _custom_process_data(self) -> None:
        pass

    def _append_system_prompt(self, row: Dict) -> Dict:
        row['system_prompt'] = 'Read the passages. Generate at most one word.'
        return row

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    dataset = LongContext(tokenizer=tokenizer)
    import pdb; pdb.set_trace()
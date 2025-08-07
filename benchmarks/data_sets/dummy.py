from datasets import load_dataset
from benchmarks_ours.data_sets.data_set import Data_set
from typing import Dict, List, Any
import logging
from transformers import AutoTokenizer
import pandas as pd
logger = logging.getLogger(__name__)

class Dummy(Data_set):

    def load_data_from_hf(self):
        return pd.DataFrame({
            'context': [[
                'Derek is living in Chrysan District.\n', 
                'All people living in Chrysan District work in Themum company.\n'
            ]],
            'input': ['Which company does Derek work in?'],
            'answers': [['Themum']],
            'length' : [100] # dummy value
        })
            
    def _split_docs(self, row: Dict) -> Dict:
        return row

    def _append_input(self, row: Dict) -> Dict:
        row['input'] = '\n\nQuestion: ' + row['input'] + '\nAnswer within 5 words:'
        return row

    def _append_system_prompt(self, row: Dict) -> Dict:
        row['system_prompt'] = 'Answer the question based on the given passages. The following are given passages.\n\n'
        return row
    
    def _custom_process_data(self) -> None:
        pass


            
if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    dataset = Dummy(tokenizer=tokenizer)
    import pdb; pdb.set_trace()
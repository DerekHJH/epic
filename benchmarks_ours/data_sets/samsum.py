from datasets import load_dataset
from benchmarks_ours.data_sets.data_set import Data_set
from typing import Dict, List, Any
import re
from transformers import AutoTokenizer

class SAMSum(Data_set):
    def load_data_from_hf(self):
       return load_dataset('THUDM/LongBench', 'samsum', split='test').to_pandas()


    def _append_input(self, row: Dict) -> Dict:
        row['input'] = '\n\nSummarize the following dialogue just like the preceding examples.\n\n' + row['input']
        return row

    def _append_system_prompt(self, row: Dict) -> Dict:
        row['system_prompt'] = 'Please summarize a dialogue. \n\nHere are some examples.\n'
        return row
    
    def _custom_process_data(self) -> None:
        pass

    def _split_docs(self, row: Dict) -> Dict:

        all_docs: str = row['context']
        delimiter = r'(Dialogue:)'
        docs: List[str] = re.split(delimiter, all_docs)
        assert docs[0] == '', f'First element of docs is not empty: {docs[0]}'
        docs = docs[1:]
        row['context'] = [docs[i] + docs[i+1] for i in range(0, len(docs), 2)]
        return row
               
if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    dataset = SAMSum(tokenizer=tokenizer)
    import pdb; pdb.set_trace()
    

        

        

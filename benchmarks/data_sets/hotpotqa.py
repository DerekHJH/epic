from datasets import load_dataset
from benchmarks_ours.data_sets.data_set import Data_set
from benchmarks_ours.data_sets.utils import qa_f1_score
from typing import Dict, List, Any
import logging
import re
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

class HotpotQA(Data_set):

    def load_data_from_hf(self):
        return load_dataset('THUDM/LongBench', 'hotpotqa', split='test').to_pandas()
            
    def _split_docs(self, row: Dict) -> Dict:
        all_docs: str = row['context']
        delimiter = r'(Passage \d+:\n)'
        docs: List[str] = re.split(delimiter, all_docs)
        assert docs[0] == '', f'First element of docs is not empty: {docs[0]}'
        docs = docs[1:]
        row['context'] = [docs[i] + docs[i+1] for i in range(0, len(docs), 2)]
        return row

    def _append_input(self, row: Dict) -> Dict:
        row['input'] = '\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: ' + row['input'] + '\nAnswer within 5 words:'
        return row

    def _append_system_prompt(self, row: Dict) -> Dict:
        row['system_prompt'] = 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n'
        return row
    
    def _custom_process_data(self) -> None:
        pass
            
if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    dataset = HotpotQA(tokenizer=tokenizer)
    import pdb; pdb.set_trace()
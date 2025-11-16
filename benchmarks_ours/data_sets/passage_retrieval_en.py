from datasets import load_dataset
from benchmarks_ours.data_sets.data_set import Data_set
from typing import Dict, List, Any
import logging
import re
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

class PassageRetrievalEn(Data_set):

    def load_data_from_hf(self):
        return load_dataset('THUDM/LongBench', 'passage_retrieval_en', split='test').to_pandas()
            
    def _split_docs(self, row: Dict) -> Dict:
        all_docs: str = row['context']
        delimiter = r'(Paragraph \d+:)'
        docs: List[str] = re.split(delimiter, all_docs)
        assert docs[0] == '', f'First element of docs is not empty: {docs[0]}'
        docs = docs[1:]
        row['context'] = [docs[i] + docs[i+1] for i in range(0, len(docs), 2)]
        return row

    def _append_input(self, row: Dict) -> Dict:
        row['input'] = '\n\nTell me which paragraph corresponds to the following summarization. Only give me the answer and do not output any other words.\n\nSummarization: ' + row['input'] + '\nAnswer in the form Paragraph x:'
        return row

    def _append_system_prompt(self, row: Dict) -> Dict:
        row['system_prompt'] = 'I will give you a summarization of a paragraph and a list of paragraphs. Tell me which paragraph in the list corresponds to the given summarization. Only give me the answer and do not output any other words.\n\nThe following are given paragraphs.\n'
        return row
    
    def _custom_process_data(self) -> None:
        pass


            
if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    dataset = PassageRetrievalEn(tokenizer=tokenizer)
    import pdb; pdb.set_trace()
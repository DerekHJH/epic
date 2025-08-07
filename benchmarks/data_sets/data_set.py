from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List
import os
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from enum import Enum
import random
from benchmarks_ours.data_sets.utils import scorer

class Data_set(ABC):
    """
    Use the underline to differentiate from huggingface 
    datasets (dataset) and Datasets (Dataset).
    """
    def __init__(
            self, 
            tokenizer,
            tot_num_data=int(1e6), 
            path: str=None,
            doc_chunk_size: int=512,
            **kwargs
        ):
        """
        Load cached dataset. If failed, load from hf in the subclass.
        """
        self.tokenizer = tokenizer
        self.tot_num_data = tot_num_data
        self.path = path
        self.doc_chunk_size = doc_chunk_size
        self.kwargs = kwargs

        self.data: pd.DataFrame = None
        if path is not None and os.path.exists(os.path.join(path, 'data.json')):
            self.data = pd.read_json(os.path.join(path, 'data.json'))
            logger.info(f'Loaded dataset from {path}')
        else:
            self.data = self.load_data_from_hf()
            logger.info('Loaded dataset from huggingface')

            # Common processing for all datasets
            self.data = self.data[:tot_num_data]
            self.data = self.data.apply(lambda row: self._split_docs(row), axis=1) \
                .apply(lambda row: self._append_input(row), axis=1) \
                .apply(lambda row: self._append_system_prompt(row), axis=1)  \
                .apply(lambda row: self._append_output_length(row), axis=1)
            
            self._custom_process_data()

    @abstractmethod
    def load_data_from_hf(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _custom_process_data(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _append_input(self, row: Dict) -> Dict:
        """
        Append the input.
        """
        raise NotImplementedError

    @abstractmethod
    def _append_system_prompt(self, row: Dict) -> Dict:
        """
        Append the system prompt.
        """
        raise NotImplementedError
    

    # ALL following methods are common to all datasets.
    # SHOULD NOT BE OVERRIDDEN.
    # This is to ensure consistent interfaces.

    def calc_accuracy(self, dataset: str, approach: str) -> None:
        """
        Compare the model output with the groundtruth and calculate the accuracy.
        Store the accuracy in the f'score_{approach}' column.

        Args:
            approach: The approach used to generate the output.

        """
        assert f"output_{approach}" in self.data.columns, f"output_{approach} not in the dataset"
        assert "answers" in self.data.columns, "answers not in the dataset"

        self.data = self.data.apply(lambda row: scorer(row, approach, dataset), axis=1)

    def update(self, new_data: Dict[str, List]) -> None:
        for key, value in new_data.items():
            if len(value) < len(self.data):
                logger.warning(f'Length of new data is less than the original data: {len(value)} < {len(self.data)}')
            self.data[key] = value + (len(self.data) - len(value)) * [None]

    def save_dataset(self, path: str) -> None:
        self.data.to_json(os.path.join(path, 'data.json'))

    def __iter__(self):
        for _, row in self.data.iterrows():
            yield row['system_prompt'], row['context'], row['input']
            
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data.iloc[idx]

    

    def _split_docs(self, row: Dict) -> Dict:
        """
        Split the context into multiple documents with fixed length
        """
        context = row['context']
        docs = []
        tokens_context = self.tokenizer.encode(context)[1:]  # Remove the first token (BOS token)
        chunk_begin = 0
        chunk_end = 0

        # We want to make sure that we cut at a sentence break so we first see what token a '.' is
        period_tokens = self.tokenizer.encode('.')
        if 842 in period_tokens: # For mistral tokenizers. mistral tokenizers have two kinds of period tokens
            period_tokens = [1, 28723]
        newline_tokens = self.tokenizer.encode('\n')

        while chunk_begin < len(tokens_context):
            chunk_end = chunk_begin + self.doc_chunk_size
            tokens_new_context = tokens_context[chunk_begin:chunk_end]

            # If we are not at the end of the context
            # We iteration forwards until we find the first period
            while chunk_end < len(tokens_context) and len(tokens_new_context) < 2*self.doc_chunk_size and \
                ((len(tokens_new_context) < 1.5*self.doc_chunk_size and \
                 tokens_new_context[-1] not in period_tokens) \
                or \
                (len(tokens_new_context) >= 1.5*self.doc_chunk_size and \
                 tokens_new_context[-1] not in newline_tokens)):
                chunk_end += 1
                tokens_new_context = tokens_context[chunk_begin:chunk_end]
            
            assert len(tokens_new_context) <= 2 * self.doc_chunk_size, f'Chunk size is too large: {tokens_new_context}'
        
            docs.append(self.tokenizer.decode(tokens_new_context))

            chunk_begin = chunk_end
            
        row['context'] = docs
        return row
        
    
    def _append_output_length(self, row: Dict) -> Dict:
        row['output_length'] = len(self.tokenizer.encode(row['answers'][0]))
        return row
        
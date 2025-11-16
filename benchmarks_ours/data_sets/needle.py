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


class Needle(Data_set):

    def load_data_from_hf(self) -> pd.DataFrame:
        """
        Construct the dataset from the haystack and needle.

        :param tokenizer: must be provided.
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param kwargs: Additional arguments.
        """
        self.retrieval_question = 'What is the best thing to do in San Francisco?'
        self.haystack_dir = "PaulGrahamEssays"
        self.needle = 'The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.'
        self.final_context_length_buffer = 100
        self.context_lengths_min = 1000
        self.context_lengths_max = 11000
        self.context_lengths_num_intervals = 20
        self.document_depth_percent_min = 0
        self.document_depth_percent_max = 100
        self.document_depth_percent_intervals = 10
        self.document_depth_percent_interval_type = 'linear' # "sigmoid",
        

        self.context_lengths = np.round(np.linspace(self.context_lengths_min, self.context_lengths_max, num=self.context_lengths_num_intervals, endpoint=True)).astype(int)
        if self.document_depth_percent_interval_type == 'linear':
            self.document_depth_percents = np.round(np.linspace(self.document_depth_percent_min, self.document_depth_percent_max, num=self.document_depth_percent_intervals, endpoint=True)).astype(int)
        elif self.document_depth_percent_interval_type == 'sigmoid':
            self.document_depth_percents = [self.logistic(x) for x in np.linspace(self.document_depth_percent_min, self.document_depth_percent_max, self.document_depth_percent_intervals)]
        
        data = []
        for context_length in tqdm(self.context_lengths):
            for depth_percent in self.document_depth_percents:
                context = self.generate_context(context_length, depth_percent)
                data.append(pd.DataFrame({
                    'context': [context], 
                    'input': [self.retrieval_question],
                    'answers': [[self.needle]],
                    'context_length': [context_length],
                    'depth_percent': [depth_percent]
                }))
        data = pd.concat(data, ignore_index=True)
        return data

    
    def logistic(self, x, L=100, x0=50, k=.1):
        if x in [0, 100]:
            return x
        x = -k * (x - x0)
        return np.round(L * self.sigmoid(x), 3)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your haystack dir files loaded into a string
        context = self.read_context_files()

        # Truncate the haystack dir essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.tokenizer.encode(self.needle)
        tokens_context = self.tokenizer.encode(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.tokenizer.encode('.')
            if 842 in period_tokens: # For mistral tokenizers. mistral tokenizers have two kinds of period tokens
                period_tokens = [1, 28723]
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.tokenizer.decode(tokens_new_context)
        return new_context

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
        row['input'] = '\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: ' + row['input'] + '\nAnswer within 20 words:'
        return row
    
    def _custom_process_data(self) -> None:
        pass

    def _append_system_prompt(self, row: Dict) -> Dict:
        row['system_prompt'] = 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n'
        return row


if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    dataset = Needle(tokenizer=tokenizer)
    import pdb; pdb.set_trace()
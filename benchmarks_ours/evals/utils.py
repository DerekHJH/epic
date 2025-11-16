
import random

import numpy as np
import torch

from benchmarks_ours.data_sets import wikimqa, musique, samsum, multi_news, hotpotqa, needle, passage_retrieval_en, long_context, dummy


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Utils for datasets
str2class = {
    '2wikimqa': wikimqa.WikiMQA,
    'musique': musique.Musique,
    'multi_news': multi_news.MultiNews,
    'hotpotqa': hotpotqa.HotpotQA,
    'samsum': samsum.SAMSum,
    '2wikimqa-beginning': wikimqa.WikiMQA,
    '2wikimqa-middle': wikimqa.WikiMQA,
    '2wikimqa-end': wikimqa.WikiMQA,
    '2wikimqa-original': wikimqa.WikiMQA, 
    'needle': needle.Needle,   
    'passage_retrieval_en': passage_retrieval_en.PassageRetrievalEn,
    'long_context': long_context.LongContext,
    'dummy': dummy.Dummy,
}


# Utils for plottings
model_names_map = {
        'mistralai/Mistral-7B-Instruct-v0.2': 'Mistral 7B Instruct',
        'meta-llama/Meta-Llama-3.1-8B-Instruct': 'Llama 3.1 8B Instruct',
        '01-ai/Yi-Coder-9B-Chat': 'Yi Coder 9B Chat'
}
dataset_names_map = {
    '2wikimqa': '2WikiMQA',
    'musique': 'MuSiQue',
    'samsum': 'SAMSum',
    'multi_news': 'MultiNews',
    'hotpotqa': 'HotpotQA',
    'needle': 'Needle'
}
dataset_metrics_map = {
    '2wikimqa': 'F1 score',
    'musique': 'F1 score',
    'samsum': 'Rouge-L score',
    'multi_news': 'Rouge-L score',
    'hotpotqa': 'F1 score',
    'needle': 'F1 score',
}

approach_names_map = {
    'fr': 'FR',
    'naive': 'Naive',
    'cacheblend-20': 'CacheBlend-20',
    'cacheblend-15': 'CacheBlend-15',
    'cacheblend-10': 'CacheBlend-10',
    'cacheblend-5': 'CacheBlend-5',
    'cacheblend-1': 'CacheBlend-1',
    'kvlink-32': 'LegoLink-32',
    'kvlink-16': 'LegoLink-16',
    'kvlink-8': 'LegoLink-8',
    'kvlink-4': 'LegoLink-4',
    'kvlink-2': 'LegoLink-2',
    'kvlink-1': 'LegoLink-1',
    'kvlink-0': 'LegoLink-0'
}
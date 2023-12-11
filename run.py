#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_dataset
import numpy as np
import torch
import os
import pandas as pd
import faiss
import time
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score


# In[ ]:


from code.py import *
import sys


# In[ ]:


if __name__ == '__main__':
    targets = sys.argv[1:]
    query = ''.join(targets)
    dataset = get_data()
    model, index = encode_semantic(dataset)
    eval_df = pd.read_csv("data/annotationStore.csv") 
    eval_Ruby = eval_df[eval_df["Language"] == "Ruby"]
    best_weights = grid_search()
    eval_metric = end_to_end(best_weights)
    query_output = search(best_weights, query)
    print("Overall Model Performance: " + str(eval_metric))
    pd.set_option('display.max_colwidth', None)
    query_output['URL'] = query_output['URL'].apply(lambda x: f'{x}')
    query_output.style.format({'URL': lambda x: x})
    print(query_output)


# Import necessary libraries
import pandas as pd
import numpy as np
import multiprocessing
import time

import pyterrier as pt
from pyterrier_dr import FlexIndex, SBertBiEncoder
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings

from transformers import pipeline



# Initialize dataset
dataset_directory = 'dataset'
ground_truth_train = pd.read_pickle(f'{dataset_directory}/ground_truth_train.pkl')
ground_truth_test = pd.read_pickle(f'{dataset_directory}/ground_truth_test.pkl')
docs = pd.read_pickle(f'{dataset_directory}/docs.pkl')

default_terrier_config_setup = {"hnsw": {
                                  "space": "cosine",
                                  "ef_construction": 100, # same as pyterrier (if increased, will increase recall but increses memory usage and indexing time)
                                  "ef_search": 100, # same as pyterrier (if increased, will increase recall but slow down search)
                                  "max_neighbors": 16, # same as pyterrier
                                  "num_threads": multiprocessing.cpu_count(),
                                  "batch_size": 100, # same as pyterrier
                                  "sync_threshold": 1000,
                                  "resize_factor": 1.2
                              }}

default_chroma_config_setup = {"hnsw": {
                                  "space": "cosine",
                                  "ef_construction": 40, # same as pyterrier (if increased, will increase recall but increses memory usage and indexing time)
                                  "ef_search": 16, # same as pyterrier (if increased, will increase recall but slow down search)
                                  "max_neighbors": 32, # same as pyterrier
                                  "num_threads": multiprocessing.cpu_count(),
                                  "batch_size": 64, # same as pyterrier
                                  "sync_threshold": 1000,
                                  "resize_factor": 1.2
                              }}


# For better efficiency on modern hardware:
qwen_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
    # tokenizer_kwargs={"padding_side": "left"},
)
# Define wrapper
class QwenEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return qwen_model.encode(input).tolist()
    

basic_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
qwen_fn = QwenEmbeddingFunction()



ids = docs['id'].tolist()
passages = docs['passage'].tolist()

test_ids = ids[:5]
test_passages = passages[:5]
basic_emb = basic_fn(test_passages)
qwen_emb = qwen_fn(test_passages)

print(type(basic_emb), type(basic_emb[0]), len(basic_emb), len(basic_emb[0]))
print(type(qwen_emb), type(qwen_emb[0]), len(qwen_emb), len(qwen_emb[0]))

df1 = pd.DataFrame({
    "id": test_ids,
    "passage": test_passages,
    "embedding": basic_emb
})

df2 = pd.DataFrame({
    "id": test_ids,
    "passage": test_passages,
    "embedding": qwen_emb
})

df1.to_csv('basic_embeddings.csv', index=False)
df2.to_csv('qwen_embeddings.csv', index=False)


mpla1 = pd.read_csv('basic_embeddings.csv')
mpla2 = pd.read_csv('qwen_embeddings.csv')

# take an embedding and check its dimensions
vec = mpla1['embedding'].iloc[0]
print(vec)
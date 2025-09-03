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

import torch
from transformers import pipeline

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Initialize dataset
dataset_directory = 'dataset'
ground_truth_train = pd.read_pickle(f'{dataset_directory}/ground_truth_train.pkl')
ground_truth_test = pd.read_pickle(f'{dataset_directory}/ground_truth_test.pkl')
docs = pd.read_pickle(f'{dataset_directory}/docs.pkl')

default_chroma_config_setup = {"hnsw": {
                                  "space": "cosine",
                                  "ef_construction": 100, # same as pyterrier (if increased, will increase recall but increses memory usage and indexing time)
                                  "ef_search": 100, # same as pyterrier (if increased, will increase recall but slow down search)
                                  "max_neighbors": 16, # same as pyterrier
                                  "num_threads": multiprocessing.cpu_count(),
                                  "batch_size": 100, # same as pyterrier
                                  "sync_threshold": 1000,
                                  "resize_factor": 1.2
                              }}

default_terrier_config_setup = {"hnsw": {
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
    model_kwargs={"attn_implementation": "flash_attention_2", 
                  "device_map": "auto", 
                  "torch_dtype": torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
    
  
)
# Define wrapper
class QwenEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
          return qwen_model.encode(input, batch_size=4).tolist()
    

def bm25_indexing(dataset):
  if not pt.java.started():
    pt.java.init()

  # PyTerrier Sparse
  indexer = pt.IterDictIndexer('./indices/new_index', text_attrs=['text'], meta={'text':1024, 'docno':20}, meta_reverse={'docno'}, overwrite=True, threads=1) # everything is default settings


  preprocessed_chunked = dataset.copy()
  preprocessed_chunked['text'] = preprocessed_chunked['text'].apply(preprocessing)
  preprocessed_chunked

  start = time.time()
  print("Starting PyTerrier indexing...")

  index_ref = indexer.index(preprocessed_chunked.to_dict(orient='records'))
  indexer.getIndexStats()

  end = time.time()
  print(f"PyTerrier Indexing completed in {(end - start)/60} minutes.")
  






def terrier_indexing(dataset):
  # PyTerrier Dense
  dense_terrier_model = SBertBiEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
  dense_terrier_index = FlexIndex(path="dense_index.flex")

  indexing_pipeline = dense_terrier_model >> dense_terrier_index.indexer(mode='overwrite')

  indexing_pipeline.index(dataset.to_dict(orient='records'))







def chroma_indexing(dataset, embed_fn):
  chroma_client = chromadb.PersistentClient(path="./chroma_db")

  # chroma_client.delete_collection("biomedical_collection_qwen") # delete previous collection if exists


  collection = chroma_client.get_or_create_collection(name="biomedical_collection_test", embedding_function=embed_fn, configuration=default_chroma_config_setup)

  docs = list(dataset['text'])
  idxs = list(dataset['docno'])

  start=time.time()
  print("Starting ChromaDB indexing...")

  batch_size = 100
  for i in range(0, len(docs), batch_size):
      collection.add(
          documents=docs[i:i+batch_size],
          ids=idxs[i:i+batch_size]
      )
      print(f"First {min(i+batch_size,len(docs))} docs have been indexed")

  end=time.time()

  print(f"ChromaDB Indexing completed in {(end - start)/60} minutes.")




def doc_splitter(doc, delimiter="."):
  chunks = []

  if delimiter == "none":
    chunks.append(doc)
    return chunks


  for chunk in doc.split(delimiter):
    if(chunk == "" or chunk.isspace()):
      continue

    chunk = chunk.strip()
    if len(chunk) < 10:
      continue

    chunks.append(chunk)

  return chunks

# Chunks the documents in the specified column and delimiter
def chunker(docs, column="body", delimiter="."): # catch potential errors

  new_rows = []

  # Run through the documents
  for _ , row in docs.iterrows():
    chunks = [chunk for chunk in doc_splitter(row[column], delimiter)]

    # Assign each chunk of the document a unique id (chunk_id)
    for i, chunk in enumerate(chunks):
      new_row = row.to_dict()
      new_row[column] = chunk

      if len(chunks) > 1:
        new_row['id'] = str(new_row['id']) + '_' + str(i)
      else:
        new_row['id'] = str(new_row['id'])


      new_rows.append(new_row)

  return pd.DataFrame(new_rows)











def preprocessing(text:str)->str:
    # To Lowercase
    text = text.lower()

    # Tokenization
    list_of_words = word_tokenize(text)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    list_of_words = [word for word in list_of_words if word not in stop_words]

    # Punctuation Removal
    list_of_words = [word for word in list_of_words if word not in string.punctuation]

    # Stemming
    stemmer = PorterStemmer()
    list_of_words = [stemmer.stem(word) for word in list_of_words]

    # Unify in a string
    text = ' '.join(list_of_words)

    return text




def main():
  chunked_df = chunker(docs, column="passage", delimiter="none")

  # Getting chunked_df in the appropriate form to feed to dense terrier pipeline
  simple_chunked = chunked_df.rename(columns={'passage': 'text', 'id': 'docno'})
  simple_chunked['docno'] = simple_chunked['docno'].astype(str)

  # Indexing
  # bm25_indexing(simple_chunked)
  # terrier_indexing(simple_chunked)

  # embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
  embed_fn = QwenEmbeddingFunction()
  chroma_indexing(simple_chunked, embed_fn)


if __name__ == "__main__":
  main()

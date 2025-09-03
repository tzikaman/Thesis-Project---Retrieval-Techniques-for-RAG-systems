# Import necessary libraries
import pandas as pd
import time

import pyterrier as pt
from pyterrier_dr import FlexIndex, SBertBiEncoder

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sentence_transformers import SentenceTransformer
import torch 

NUM_RESULTS = 500

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

"""##Terrier Sparse Retrieval"""
def run_bm25(template_query):
    if not pt.java.started():
        pt.java.init()

    bm25_query = template_query.copy()
    bm25_query['query'] = bm25_query['query'].apply(preprocessing)
    bm25_query['query'] = bm25_query['query'].str.replace("/", " ")
    bm25_query['query'] = bm25_query['query'].str.replace("'", " ")
    bm25_query['query'] = bm25_query['query'].str.replace('?', " ")
    bm25_query['query'] = bm25_query['query'].str.replace('"', " ")
    bm25_query['query'] = bm25_query['query'].str.replace('(', " ")
    bm25_query['query'] = bm25_query['query'].str.replace(')', " ")
    bm25_query['query'] = bm25_query['query'].str.replace('!', " ")
    bm25_query['query'] = bm25_query['query'].str.replace(',', " ")
    bm25_query['query'] = bm25_query['query'].str.replace('.', " ")
    bm25_query['query'] = bm25_query['query'].str.replace(';', " ")
    bm25_query['query'] = bm25_query['query'].str.replace(':', " ")
    bm25_query['query'] = bm25_query['query'].str.replace('  ', " ")



    index = pt.IndexFactory.of("./indices/new_index")
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=NUM_RESULTS)




    print("\n\nStarting bm25 Retrieval...")
    start = time.time()
    bm25_results = bm25.transform(bm25_query)
    end = time.time()
    print(f"BM25 Retrieval completed in {(end - start)} seconds.")

    print(bm25_results.info())


    bm25_results.to_csv('retrieval_results/bm25_results.csv', index=False)

    return bm25_results



"""##Dense Retrieval

###Terrier Dense Retrieval
"""
def run_terrier(template_query):
    dense_terrier_model = SBertBiEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dense_terrier_index = FlexIndex(path="dense_index.flex")
    dense_terrier_retriever = dense_terrier_index.faiss_hnsw_retriever(num_results=NUM_RESULTS, 
                                                                    neighbours=16, 
                                                                    ef_construction=100, 
                                                                    ef_search=100, 
                                                                    qbatch=100, 
                                                                    cache=True, 
                                                                    search_bounded_queue=True, 
                                                                    drop_query_vec=False)

    retrieval_pipeline = dense_terrier_model >> dense_terrier_retriever

    print("\n\nStarting Dense Terrier Retrieval...")
    start = time.time()
    terrier_results = retrieval_pipeline.transform(template_query)
    end = time.time()
    print(f"Dense Terrier Retrieval completed in {(end - start)} seconds.")

    print(terrier_results.info())


    terrier_results.drop(columns=['query_vec'], inplace=True)
    terrier_results.to_csv('retrieval_results/dense_terrier_results_maybe_better.csv', index=False)

    return terrier_results




# Chroma Retrieval
def run_chroma(template_query):
    qwen_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
    # tokenizer_kwargs={"padding_side": "left"},
    )
    # Define wrapper
        
    class QwenQueryEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return qwen_model.encode(input, batch_size=4, prompt_name='query').tolist()
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="biomedical_collection_test", embedding_function=QwenQueryEmbeddingFunction())


    print("\n\nStarting ChromaDB Retrieval...")
    start = time.time()
    # Build records directly instead of merging dicts then flattening
    records = []
    batch_size = 20
    for i in range(0, len(template_query), batch_size):
        batch_df = template_query.iloc[i:i+batch_size]
        res = collection.query(
            query_texts=batch_df['query'].tolist(),
            n_results=NUM_RESULTS
        )
        # res['ids'] and res['distances'] are lists (per query) of lists
        for local_idx, row in enumerate(batch_df.itertuples()):
            doc_ids = res['ids'][local_idx]
            distances = res['distances'][local_idx]
            for rank, (docno, score) in enumerate(zip(doc_ids, distances)):
                records.append({
                    'qid': row.qid,
                    'query': row.query,
                    'docno': docno,
                    'score': score,  # distance (lower is better); keep name for parity
                    'rank': rank
                })

    chroma_results = pd.DataFrame(records)
    end = time.time()
    print(f"ChromaDB Retrieval completed in {(end - start)} seconds.")
    print(chroma_results.info())


    chroma_results.to_csv('retrieval_results/chroma_results_test.csv', index=False)

    return chroma_results

def run():
    # Initialize dataset
    dataset_directory = 'dataset'
    ground_truth = pd.read_pickle(f'{dataset_directory}/ground_truth_train.pkl')
    docs = pd.read_pickle(f'{dataset_directory}/docs.pkl')


    ##Setting the query

    template_query = ground_truth[['id', 'question']].rename(columns={'id': 'qid', 'question': 'query'})
    template_query['qid'] = template_query['qid'].astype(str)

    # run_bm25(template_query)
    # run_terrier(template_query)
    run_chroma(template_query)


if __name__ == "__main__":
    run()
import pandas as pd
from sentence_transformers import CrossEncoder

docs = pd.read_pickle('dataset/docs.pkl')
ground_truth = pd.read_pickle('dataset/ground_truth_train.pkl')

bm25_results = pd.read_csv('retrieval_results/bm25_results.csv')
# terrier_results = pd.read_csv('dense_terrier_results.csv')
chroma_results = pd.read_csv('retrieval_results/chroma_results_test.csv')

bm25_groups = bm25_results.groupby('qid', sort=False)
# terrier_groups = terrier_results.groupby('qid', sort=False)
chroma_groups = chroma_results.groupby('qid', sort=False)

# hybrid_1_results = pd.DataFrame()
# hybrid_2_results = pd.DataFrame()
hybrid_3_results = pd.DataFrame()

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
count = 0
for name, group in bm25_results.groupby('qid', sort=False):

    # Get top 100 doc ids from each retrieval method
    bm25_retrieved_docs = group['docno'].tolist()[:100]
    
    # terrier_retrieved_docs = terrier_groups.get_group(name)['docno'].tolist() if name in terrier_groups.groups else []
    # terrier_retrieved_docs = terrier_retrieved_docs[:100]

    chroma_retrieved_docs = chroma_groups.get_group(name)['docno'].tolist() if name in chroma_groups.groups else []
    chroma_retrieved_docs = chroma_retrieved_docs[:100]

    # Combine them and prepare them for the reranker
    # bm_terrier_combined = set(bm25_retrieved_docs).union(set(terrier_retrieved_docs))
    bm_chroma_combined = set(bm25_retrieved_docs).union(set(chroma_retrieved_docs))
    
    # pairs_1 = [(ground_truth.loc[ground_truth['id']==name, 'question'].iloc[0], docs.loc[docs['id']==docno, 'passage'].iloc[0]) for docno in bm_terrier_combined]
    pairs_2 = [(ground_truth.loc[ground_truth['id']==name, 'question'].iloc[0], docs.loc[docs['id']==docno, 'passage'].iloc[0]) for docno in bm_chroma_combined]

    # Rerank them
    # scores_1 = reranker.predict(pairs_1)
    scores_2 = reranker.predict(pairs_2)

    # df_1 = pd.DataFrame({
    #     'qid': name,
    #     'docno': list(bm_terrier_combined),
    #     'score': scores_1
    # })

    # df_1.explode(['docno','score']).reset_index(drop=True)
    # hybrid_1_results = pd.concat([hybrid_1_results, df_1], axis=0, ignore_index=True)



    df_2 = pd.DataFrame({
        'qid': name,
        'docno': list(bm_chroma_combined),
        'score': scores_2
    })

    df_2.explode(['docno', 'score']).reset_index(drop=True)
    hybrid_3_results = pd.concat([hybrid_3_results, df_2], axis=0, ignore_index=True)

    print(f"Processed query {count+1}/{len(ground_truth)}")
    count += 1

# sort dfs by score by group in descending order
# hybrid_1_results = hybrid_1_results.sort_values(by=['qid', 'score'], ascending=[True, False])
# hybrid_2_results = hybrid_2_results.sort_values(by=['qid', 'score'], ascending=[True, False])
hybrid_3_results = hybrid_3_results.sort_values(by=['qid', 'score'], ascending=[True, False])

# hybrid_1_results.to_csv('hybrid_1_results.csv')
# hybrid_2_results.to_csv('hybrid_2_results.csv')
hybrid_3_results.to_csv('hybrid_3_results.csv')
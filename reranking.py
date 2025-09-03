import pandas as pd
from sentence_transformers import CrossEncoder
import torch

directory = "retrieval_results"

docs = pd.read_pickle('dataset/docs.pkl')
ground_truth = pd.read_pickle('dataset/ground_truth_train.pkl')

# bm25_results = pd.read_csv(f'{directory}/bm25_results.csv')
chroma_results = pd.read_csv(f'{directory}/chroma_results_test.csv')

# bm25_groups = bm25_results.groupby('qid', sort=False)
chroma_groups = chroma_results.groupby('qid', sort=False)

# bm25_reranked = pd.DataFrame()
chroma_test_reranked = pd.DataFrame()

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2',
                        device='cuda' if torch.cuda.is_available() else 'cpu',)

count = 0
for name, group in chroma_results.groupby('qid', sort=False):

    # Get top 100 doc ids from each retrieval method
    chroma_retrieved_docs = group['docno'].tolist()[:100]

    # bm25_retrieved_docs = bm25_groups.get_group(name)['docno'].tolist() if name in bm25_groups.groups else []
    # bm25_retrieved_docs = bm25_retrieved_docs[:100]

    # pairs_bm25 = [(ground_truth.loc[ground_truth['id']==name, 'question'].iloc[0], docs.loc[docs['id']==docno, 'passage'].iloc[0]) for docno in bm25_retrieved_docs]
    pairs_chroma = [(ground_truth.loc[ground_truth['id']==name, 'question'].iloc[0], docs.loc[docs['id']==docno, 'passage'].iloc[0]) for docno in chroma_retrieved_docs]

    # Rerank them
    # scores_bm25 = reranker.predict(pairs_bm25)
    scores_chroma = reranker.predict(pairs_chroma)

    # df_1 = pd.DataFrame({
    #     'qid': name,
    #     'docno': list(bm25_retrieved_docs),
    #     'score': scores_bm25
    # })

    # df_1.explode(['docno','score']).reset_index(drop=True)
    # bm25_reranked = pd.concat([bm25_reranked, df_1], axis=0, ignore_index=True)

    df_2 = pd.DataFrame({
        'qid': name,
        'docno': list(chroma_retrieved_docs),
        'score': scores_chroma
    })

    df_2.explode(['docno', 'score']).reset_index(drop=True)
    chroma_test_reranked = pd.concat([chroma_test_reranked, df_2], axis=0, ignore_index=True)

    if (count+1) % 500 == 0:
        print(f"Processed query {count+1}/{len(ground_truth)}")
    count += 1


# sort dfs by score by group in descending order
# bm25_reranked = bm25_reranked.sort_values(by=['qid', 'score'], ascending=[True, False])
chroma_test_reranked = chroma_test_reranked.sort_values(by=['qid', 'score'], ascending=[True, False])

# bm25_reranked.to_csv(f'{directory}/bm25_reranked.csv')
chroma_test_reranked.to_csv(f'{directory}/chroma_test_reranked.csv')


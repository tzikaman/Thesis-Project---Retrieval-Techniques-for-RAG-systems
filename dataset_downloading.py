import pandas as pd

splits = {'train': 'question-answer-passages/train-00000-of-00001.parquet', 'test': 'question-answer-passages/test-00000-of-00001.parquet'}
ground_truth_train = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/" + splits["train"])
ground_truth_test = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/" + splits["test"])

docs = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/text-corpus/test-00000-of-00001.parquet")

# Save the datasets locally
directory = 'dataset'
ground_truth_train.to_pickle(f'{directory}/ground_truth_train.pkl')
ground_truth_test.to_pickle(f'{directory}/ground_truth_test.pkl')
docs.to_pickle(f'{directory}/docs.pkl')

# Print confirmation
print("Datasets have been downloaded and saved locally.")
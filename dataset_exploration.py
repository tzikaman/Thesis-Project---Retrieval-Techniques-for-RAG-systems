import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import dataset from web 
# splits = {'train': 'question-answer-passages/train-00000-of-00001.parquet', 'test': 'question-answer-passages/test-00000-of-00001.parquet'}
# ground_truth_train = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/" + splits["train"])
# ground_truth_test = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/" + splits["test"])

# docs = pd.read_parquet("hf://datasets/enelpol/rag-mini-bioasq/text-corpus/test-00000-of-00001.parquet")


# Import dataset from local files
dataset_directory = 'dataset'
ground_truth_train = pd.read_pickle(f'{dataset_directory}/ground_truth_train.pkl')
ground_truth_test = pd.read_pickle(f'{dataset_directory}/ground_truth_test.pkl')
docs = pd.read_pickle(f'{dataset_directory}/docs.pkl')


# Augment ground truth with a column about the length of the
# list of the relevant passage ids
ground_truth_train['list_lengths'] = ground_truth_train['relevant_passage_ids'].str.len()
ground_truth_test['list_lengths'] = ground_truth_test['relevant_passage_ids'].str.len()


# Augment documents with a column about the # of words
docs['num_words'] = docs['passage'].str.split().str.len()


# Extract some valuable information from data
print(f"Train data:\n{ground_truth_train.head()['relevant_passage_ids']}\n")

print(f"Test data:\n{ground_truth_test.info()}\n")

print(f"Passages:\n{docs.info()}")


# Plot the frequencies
fig, axes = plt.subplots(2, 1, figsize=(12,6))


len_counts = ground_truth_train['list_lengths'].value_counts().sort_index()

len_counts.plot(kind='bar', ax=axes[0], grid=True)
axes[0].set_xlabel('Length of List')
axes[0].set_ylabel('# of Lists')
axes[0].set_title('Frequency of List Lengths')



docs['num_words'].hist(bins=200, ax=axes[1], grid=True, color='orange')
axes[1].set_xlabel('# of Words')
axes[1].set_ylabel('# of Documents')
axes[1].set_title('Distribution of Words in a List')
axes[1].set_xticks(np.linspace(docs['num_words'].min(),
                           docs['num_words'].max(),
                           num=20))

plt.tight_layout()
plt.show()
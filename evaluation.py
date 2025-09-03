import os
import pandas as pd
import matplotlib.pyplot as plt
import cProfile
import numpy as np
import itertools


def ensure_output_dir(directory):
    os.makedirs(directory, exist_ok=True)


def evaluate(df: pd.DataFrame, relevant_passages: dict, k: int = 10) -> pd.DataFrame:
    """
    Evaluate the model's performance using precision@k and recall@k metrics.
    """
    results = {}

    for qid, group in df.groupby("qid", sort=False):
        top_k = group["docno"].iloc[:k].to_list()

        relevant = relevant_passages.get(qid, set())
        # hits = len(set(top_k) & relevant)
        hits = [1 if doc in relevant else 0 for doc in top_k]

        # Precision@k
        precision = sum(hits) / k if k else 0.0

        # Recall@k
        recall = sum(hits) / len(relevant) if relevant else 0.0
        # f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0

        # MRR@k
        try:
            mrr = 1 / (hits.index(1) + 1)
        except ValueError:
            mrr = 0.0

        # MAP@k
        precisions_at_hits = [sum(hits[:i+1]) / (i+1) for i, h in enumerate(hits) if h == 1]
        map = np.mean(precisions_at_hits) if precisions_at_hits else 0.0

        # nDCG@k



        results[qid] = {
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"mrr@{k}": mrr,
            f"map@{k}": map,
            # f"ndcg@{k}": hits / np.log2(np.arange(2, hits + 2)).sum() if hits > 0 else 0.0,
            # f"f2_score@{k}": f2
        }

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "qid"
    return df


def evaluate_method(df: pd.DataFrame, relevant_passages: dict, output_prefix: str) -> pd.DataFrame:
    """
    Evaluate the model's performance using precision@k and recall@k metrics for multiple k-values.
    """

    # Evaluate at multiple k-values (precision@k, recall@k) and combine results
    for k in [500, 100, 50, 20, 10, 5, 3, 1]:
        res_k = evaluate(df, relevant_passages, k=k)

        if k == 500:
            res = res_k
        else:
            res = res.merge(res_k, on='qid')

    # Print and save results
    print(f"{output_prefix} Evaluation Results:")
    print(f"{res.describe().loc['mean']}\n")

    res.to_csv(f'evaluation_results/{output_prefix}_evaluation.csv')

    # Plot results
    # metrics = res.columns
    # means = res.describe().loc['mean']
    # stds = res.describe().loc['std']

    # plt.errorbar(metrics, means, yerr=stds, capsize=5, fmt='o', markersize=8, color='skyblue')
    # plt.ylabel('Value')
    # plt.xticks(rotation=90)
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.title(f'{output_prefix} Evaluation Metrics')
    # plt.tight_layout()
    # plt.savefig(f'evaluation_results/figs/{output_prefix}_evaluation_metrics.png')
    # plt.clf()

    return res  


def color_generator():
    basic_colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]
    tab10_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    yield from itertools.cycle(tab10_colors)


def plot_eval_metric(method_name, x, y, color, yerr=None):
    plt.errorbar(x, y, yerr=yerr, capsize=5, fmt='-o', markersize=8, label=method_name, color=color)
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize="5", framealpha=0.2)

def plot_set_metrics(methods, metrics, prefix):
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        gen = color_generator()

        for i, (method_name, res) in enumerate(methods.items()):
            df = res.filter(like=metric)

            metric_levels = df.columns
            x = np.arange(len(metric_levels))
            # offset = (i - len(methods)/2) * 0.1

            stats = df.describe()
            means = stats.loc['mean']
            stds = stats.loc['std']

            plot_eval_metric(method_name, x=x, y=means, yerr=stds, color=next(gen))
            plt.title(f'{metric.capitalize()} {prefix} Cumulative Plot')
        plt.xticks(x, metric_levels)
        plt.savefig(f'evaluation_results/figs/{prefix}_Comparison_{metric}.png', dpi=300)
        plt.tight_layout()
        plt.clf()

def main():
    dataset_directory = 'dataset'
    retrieval_results_dir = 'retrieval_results'
    ground_truth = pd.read_pickle(f'{dataset_directory}/ground_truth_train.pkl')
    ensure_output_dir('evaluation_results')
    relevant_passages = dict(zip(ground_truth['id'], ground_truth['relevant_passage_ids'].apply(set)))

    metrics = ['precision', 'recall', 'mrr', 'map']
    basic_methods = {
        'bm25': pd.read_csv(f'{retrieval_results_dir}/bm25_results.csv'),
        'terrier': pd.read_csv(f'{retrieval_results_dir}/dense_terrier_results.csv'),
        'terrier_maybe_better': pd.read_csv(f'{retrieval_results_dir}/dense_terrier_results_maybe_better.csv'),
        'chroma': pd.read_csv(f'{retrieval_results_dir}/chroma_results.csv'),
        'chroma_maybe_better': pd.read_csv(f'{retrieval_results_dir}/chroma_results_maybe_better.csv'),
        'chroma_qwen': pd.read_csv(f'{retrieval_results_dir}/chroma_results_qwen.csv'),

    }

    complicated_methods = {
        'bm25_reranked': pd.read_csv(f'{retrieval_results_dir}/bm25_reranked.csv'),
        'chroma_maybe_better_reranked': pd.read_csv(f'{retrieval_results_dir}/chroma_maybe_better_reranked.csv'),
        'chroma_test': pd.read_csv(f'{retrieval_results_dir}/chroma_test_reranked.csv'),
        'hybrid_1': pd.read_csv(f'{retrieval_results_dir}/hybrid_1_results.csv'),
        'hybrid_2': pd.read_csv(f'{retrieval_results_dir}/hybrid_2_results.csv'),
        'hybrid_3': pd.read_csv(f'{retrieval_results_dir}/hybrid_3_results.csv')
    }

    basic_eval_results = {}
    for name, df in basic_methods.items():
        basic_eval_results[name] = evaluate_method(df, relevant_passages, name)

    complicated_eval_results = {}
    for name, df in complicated_methods.items():
        complicated_eval_results[name] = evaluate_method(df, relevant_passages, name)

    
    plot_set_metrics(basic_eval_results, metrics, prefix='Basic')
    plot_set_metrics(complicated_eval_results, metrics, prefix='Complex')
    

    



if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    main()  # run your code

    profiler.disable()
    profiler.dump_stats("eval_stats.prof")  # save results

    stats = pstats.Stats("eval_stats.prof")
    stats.strip_dirs()
    stats.sort_stats("cumtime").print_stats(10)  # top 10 slow functions


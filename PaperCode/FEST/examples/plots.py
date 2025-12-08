import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from privacy_utility_framework.privacy_utility_framework.dataset.dataset import DatasetManager
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.basic_stats import \
    BasicStatsCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.ks_test import KSCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.wasserstein import \
    WassersteinMethod, WassersteinCalculator
from privacy_utility_framework.privacy_utility_framework.plots.plots import plot_original_vs_synthetic, \
    mutual_information_heatmap, plot_pairwise_relationships, correlation_plot_heatmap, plot_all_stats_for_stat

def wasserstein_plot_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes"]
    all_wasserstein_distances = {method: [] for method in WassersteinMethod}
    print(all_wasserstein_distances)
    for orig in original_datasets:
        for syn in synthetic_datasets:
            print(f"~~~PAIR: {orig, syn}~~~")
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            for method in all_wasserstein_distances:
                calc = WassersteinCalculator(original_data, synthetic_data)
                res = calc.evaluate(metric=method)
                all_wasserstein_distances[method].append(res)

        # Plot the distances
        bar_width = 0.35  # the width of the bars
        index = np.arange(len(synthetic_datasets))
        for i, method in enumerate(WassersteinMethod):
            plt.bar(index + i * bar_width, all_wasserstein_distances[method], bar_width, label=method.value)

        plt.xlabel('Synthetic Datasets')
        plt.ylabel('Distance')
        plt.title(f'Wasserstein, Wasserstein Sample and Sinkhorn Distances for Different Synthetic Datasets (Original: {orig})')
        plt.xticks(index + bar_width / 2, synthetic_datasets)
        plt.legend()
        plt.ylim(0, max([max(vals) for vals in all_wasserstein_distances.values()]) * 1.1)  # Adjust y-axis limit for better visualization

        plt.show()
def mutual_information_plot_example():
    original_data = pd.read_csv('../datasets/original/diabetes.csv')
    synthetic_data = pd.read_csv('../datasets/synthetic/diabetes_datasets/ctgan_sample.csv')

    mutual_information_heatmap(original_data, synthetic_data, f"diabetes_ctgan_pairwise_norm_mi.png", "dia", "ctgan")

def pairwise_plot_example():
    original_data = pd.read_csv('../datasets/original/diabetes.csv')
    synthetic_data = pd.read_csv('../datasets/synthetic/diabetes_datasets/ctgan_sample.csv')
    manager = DatasetManager(original_data, synthetic_data)

    # Set the transformer and scaler for the datasets
    manager.set_transformer_and_scaler_for_datasets()

    # Transform and normalize the datasets
    manager.transform_and_normalize_datasets()

    plot_pairwise_relationships(manager.original_dataset.transformed_normalized_data, manager.synthetic_dataset.transformed_normalized_data, 'Pairwise Relationships: Original vs Synthetic Data')

def plot_attributes_example():
    original_data = pd.read_csv('../datasets/original/diabetes.csv')
    synthetic_data = pd.read_csv('../datasets/synthetic/diabetes_datasets/ctgan_sample.csv')

    manager = DatasetManager(original_data, synthetic_data)

    # Set the transformer and scaler for the datasets
    manager.set_transformer_and_scaler_for_datasets()

    # Transform and normalize the datasets
    manager.transform_and_normalize_datasets()

    plot_original_vs_synthetic(manager.original_dataset.transformed_normalized_data, manager.synthetic_dataset.transformed_normalized_data)

def ks_test_plot_comparison():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance", "cardio", "diabetes"]
    for orig in original_datasets:
        similarities = []
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            calc = KSCalculator(original_data, synthetic_data)
            result = calc.evaluate()
            similarities.append(result)

        print(f"Dataset {orig} with KS-Similarity for each synthetic dataset: copulagan, ctgan, gaussian_copula, gmm, tvae, random")
        print(similarities)
        plt.figure(figsize=(10, 5))
        plt.bar(synthetic_datasets, similarities, color='skyblue')
        plt.xlabel('Synthetic Datasets')
        plt.ylabel('KS Similarity')
        plt.title(f'Mean KS Similarity for Different Synthetic Datasets (Original: {orig})')
        plt.ylim(0, 1)  #
        plt.show()

def correlation_plot_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "cardio", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            correlation_plot_heatmap(original_data, synthetic_data, original_name=orig, synthetic_name=syn)


def basic_stats_plot_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "cardio", "insurance"]

    for orig in original_datasets:
        all_stats = {}
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            calc = BasicStatsCalculator(original_data, synthetic_data)
            print(f"PAIR {orig, syn}")
            all_stats[f'{orig}_{syn}'] = calc.compute_basic_stats()
        for stat in ['mean', 'median', 'var']:
            plot_all_stats_for_stat(all_stats, stat, orig)
            
            
if __name__ == "__main__":
    mutual_information_plot_example()
    pairwise_plot_example()
    plot_attributes_example()
    ks_test_plot_comparison()
    correlation_plot_example()
    wasserstein_plot_example()
    basic_stats_plot_example()
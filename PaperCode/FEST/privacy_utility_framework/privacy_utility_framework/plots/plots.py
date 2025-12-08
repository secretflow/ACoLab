from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.correlation import \
    CorrelationCalculator, CorrelationMethod
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.mutual_information import \
    MICalculator


# Function to plot the distributions of each column in the original and synthetic datasets side by side
def plot_original_vs_synthetic(original_data, synthetic_data):
    num_columns = len(original_data.columns)
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5 * num_columns))

    # Plot each column’s distribution for both original and synthetic datasets
    for i, column in enumerate(original_data.columns):
        sns.kdeplot(original_data[column], ax=axes[i], label='Original', color='blue')
        sns.kdeplot(synthetic_data[column], ax=axes[i], label='Synthetic', color='red')
        axes[i].set_title(f'Distribution of {column}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


# Function to visualize pairwise relationships between columns in both original and synthetic datasets
def plot_pairwise_relationships(original_data, synthetic_data, title):
    # Combine the datasets for comparison
    original_data['Dataset'] = 'Original'
    synthetic_data['Dataset'] = 'Synthetic'
    combined_data = pd.concat([original_data, synthetic_data])

    # Create pairplot
    sns.pairplot(combined_data, hue='Dataset', plot_kws={'alpha': 0.5})
    plt.suptitle(title, y=1.02)
    plt.show()


# Function to create a heatmap showing pairwise mutual information (MI) for both original and synthetic datasets
def mutual_information_heatmap(original_data, synthetic_data, figure_filepath, orig, syn, attributes: List = None):
    # If attributes are specified, limit dataframes to those columns
    if attributes:
        orig_df = original_data[attributes]
        syn_df = synthetic_data[attributes]
    else:
        orig_df = original_data
        syn_df = synthetic_data

    # Calculate pairwise mutual information for original and synthetic data
    private_mi = MICalculator.pairwise_attributes_mutual_information(orig_df)
    synthetic_mi = MICalculator.pairwise_attributes_mutual_information(syn_df)

    # Plot heatmaps for both original and synthetic MI matrices
    fig = plt.figure(figsize=(15, 6), dpi=120)
    fig.suptitle('Pairwise Mutual Information Comparison (Original vs Synthetic)', fontsize=20)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    sns.heatmap(private_mi, ax=ax1, cmap="GnBu")
    sns.heatmap(synthetic_mi, ax=ax2, cmap="GnBu")
    ax1.set_title(f'Original Data NMI, {orig} dataset', fontsize=15)
    ax2.set_title(f'Synthetic Data NMI, generated with {syn}', fontsize=15)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.subplots_adjust(top=0.83)
    plt.savefig(figure_filepath, bbox_inches='tight')
    plt.show()


# Function to plot a heatmap comparing correlations between original and synthetic datasets
def correlation_plot_heatmap(original_data, synthetic_data, original_name, synthetic_name):
    calc = CorrelationCalculator(original_data, synthetic_data, original_name=original_name,
                                 synthetic_name=synthetic_name)
    method = CorrelationMethod.PEARSON
    orig_corr, syn_corr = calc.correlation_pairs(method)
    fig = plt.figure(figsize=(15, 6), dpi=120)
    fig.suptitle(f'{method} Correlation Comparison (Original vs Synthetic)', fontsize=20)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    sns.heatmap(orig_corr, ax=ax1, cmap="GnBu")
    sns.heatmap(syn_corr, ax=ax2, cmap="GnBu")
    ax1.set_title(f'Original Data Correlation, {calc.original.name} dataset', fontsize=15)
    ax2.set_title(f'Synthetic Data Correlation, generated with {calc.synthetic.name}', fontsize=15)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.subplots_adjust(top=0.83)
    plt.show()


# Function to plot basic statistical metrics (e.g., mean, standard deviation) of original vs synthetic datasets
def plot_all_stats_for_stat(all_stats, stat, orig):
    plt.figure(figsize=(12, 6))

    # Plot each synthetic dataset’s values for the specified statistic
    for label, stats in all_stats.items():
        syn_values = [values[f'syn_{stat}'] for values in stats.values()]
        columns = stats.keys()

        plt.plot(columns, syn_values, label=label, marker='o')

    # Plot the original dataset values for the same statistic
    first_label = list(all_stats.keys())[0]
    orig_values = [all_stats[first_label][col][f'orig_{stat}'] for col in columns]
    plt.plot(columns, orig_values, label='Original', marker='x', linestyle='--', linewidth=2)

    plt.xlabel('Columns')
    plt.ylabel(stat.capitalize())
    plt.title(f'Comparison of {stat.capitalize()} for Original ({orig}) and Synthetic Datasets')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{orig}_{stat.capitalize()}_basic_stats.png", bbox_inches='tight')
    plt.show()

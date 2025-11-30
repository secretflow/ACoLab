import numpy as np
import pandas as pd
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics import UtilityMetricCalculator


class BasicStatsCalculator(UtilityMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 original_name: str = None,
                 synthetic_name: str = None):
        """
        Initializes the BasicStatsCalculator with original and synthetic datasets.

        Parameters:
        - original: pd.DataFrame; the original dataset.
        - synthetic: pd.DataFrame; the synthetic dataset generated for analysis.
        - original_name: str (default: None); the name of the original dataset for reporting.
        - synthetic_name: str (default: None); the name of the synthetic dataset for reporting.
        """
        super().__init__(original, synthetic, original_name=original_name, synthetic_name=synthetic_name)

    def _calculate_score(self, all_stats, stat_type):
        """
        Calculate the average absolute difference for a specified statistic type (mean, median, or variance)
        between the original and synthetic datasets.

        Parameters:
        - all_stats: dict; a dictionary containing statistics for both datasets.
        - stat_type: str; the type of statistic to calculate ('mean', 'median', or 'var').

        Returns:
        - float; the average absolute difference for the specified statistic type.
        """
        diffs = []
        for key in all_stats:
            for col, stats in all_stats[key].items():
                if stat_type == 'mean':
                    diffs.append(abs(stats['syn_mean'] - stats['orig_mean']))
                elif stat_type == 'median':
                    diffs.append(abs(stats['syn_median'] - stats['orig_median']))
                elif stat_type == 'var':
                    diffs.append(abs(stats['syn_var'] - stats['orig_var']))
        return np.mean(diffs)

    def compute_basic_stats(self):
        """
        Compute the mean, median, and variance for each column in the original and synthetic datasets.

        Returns:
        - dict; a dictionary containing mean, median, and variance for each column in both datasets.
        """
        # Retrieve transformed and normalized data from original and synthetic datasets
        original = self.original.transformed_normalized_data
        synthetic = self.synthetic.transformed_normalized_data
        stats = {}
        for col in original.columns:
            # Compute statistics for the original and synthetic data
            orig_mean, syn_mean = original[col].mean(), synthetic[col].mean()
            orig_median, syn_median = original[col].median(), synthetic[col].median()
            orig_var, syn_var = original[col].var(), synthetic[col].var()

            # Store statistics in a dictionary for each column
            stats[col] = {
                'orig_mean': orig_mean, 'syn_mean': syn_mean,
                'orig_median': orig_median, 'syn_median': syn_median,
                'orig_var': orig_var, 'syn_var': syn_var
            }
        return stats

    def evaluate(self):
        """
        Evaluate the similarity between the original and synthetic datasets by calculating
        the mean absolute differences for mean, median, and variance.

        Returns:
        - dict; a dictionary with the average score for each statistic type ('mean', 'median', 'var').
        """
        # Compute basic statistics and store in a dictionary
        all_stats = {f'{self.original.name}_{self.synthetic.name}': self.compute_basic_stats()}
        scores = {}
        # Calculate final scores for mean, median, and variance
        for stat_type in ['mean', 'median', 'var']:
            final_score = self._calculate_score(all_stats, stat_type)
            scores[stat_type] = final_score
        return scores

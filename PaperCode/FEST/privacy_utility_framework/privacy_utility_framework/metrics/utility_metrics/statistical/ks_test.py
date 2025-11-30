import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics import UtilityMetricCalculator


class KSCalculator(UtilityMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 original_name: str = None,
                 synthetic_name: str = None):
        """
        Initializes the KSCalculator with original and synthetic datasets.

        Parameters:
        - original: pd.DataFrame; the original dataset.
        - synthetic: pd.DataFrame; the synthetic dataset generated for analysis.
        - original_name: str (default: None); the name of the original dataset for reporting.
        - synthetic_name: str (default: None); the name of the synthetic dataset for reporting.
        """
        super().__init__(original, synthetic, original_name=original_name, synthetic_name=synthetic_name)

    def evaluate(self):
        """
        Calculates the mean Kolmogorov-Smirnov (KS) similarity score between the original and synthetic datasets.

        Returns:
        - float; the mean KS similarity score, with higher values indicating greater similarity.
        """
        # Run KS tests on each feature
        ks_results = self.ks_test_columns()
        # Calculate the mean KS similarity score based on the KS statistic
        mean_ks_similarity = np.mean([1 - result['KS Statistic'] for result in ks_results.values()])
        return mean_ks_similarity

    def ks_test_columns(self):
            """
            Perform the Kolmogorov-Smirnov (KS) test on each feature of the original and synthetic datasets.

            Returns:
            - dict; a dictionary with feature names as keys and KS statistics and p-values as values.
            """
            # Retrieve transformed and normalized data from original and synthetic datasets
            original = self.original.transformed_normalized_data
            synthetic = self.synthetic.transformed_normalized_data
            ks_results = {}

            # Perform KS test for each feature in the dataset
            for col in original.columns:
                # Calculate KS statistic and p-value for each column
                ks_stat, p_value = ks_2samp(original[col], synthetic[col])
                ks_results[col] = {'KS Statistic': ks_stat, 'p-value': p_value}
            return ks_results

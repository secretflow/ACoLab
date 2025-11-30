from enum import Enum
import numpy as np
import pandas as pd

from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics import UtilityMetricCalculator


class CorrelationMethod(Enum):
    # Enum for defining correlation methods
    PEARSON = "pearson"  # Pearson correlation method
    SPEARMAN = "spearman"  # Spearman correlation method


class CorrelationCalculator(UtilityMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 original_name: str = None,
                 synthetic_name: str = None):
        """
        Initializes the CorrelationCalculator with original and synthetic datasets.

        Parameters:
        - original: pd.DataFrame; the original dataset.
        - synthetic: pd.DataFrame; the synthetic dataset generated for analysis.
        - original_name: str (default: None); the name of the original dataset for reporting.
        - synthetic_name: str (default: None); the name of the synthetic dataset for reporting.
        """
        super().__init__(original, synthetic, original_name=original_name, synthetic_name=synthetic_name)

    def correlation_pairs(self, method):
        """
        Calculate pairwise correlation matrices for the original and synthetic datasets
        based on the specified correlation method.

        Parameters:
        - method: CorrelationMethod; the correlation method to be applied (Pearson or Spearman).

        Returns:
        - orig_corr: pd.DataFrame; correlation matrix for the original dataset.
        - syn_corr: pd.DataFrame; correlation matrix for the synthetic dataset.
        """
        # Retrieve transformed and normalized data from original and synthetic datasets
        original = self.original.transformed_normalized_data
        synthetic = self.synthetic.transformed_normalized_data

        # Calculate correlation matrices for the original and synthetic datasets
        orig_corr = original.corr(method=method.value)
        syn_corr = synthetic.corr(method=method.value)
        return orig_corr, syn_corr

    def evaluate(self, method=CorrelationMethod.PEARSON):
        """
        Evaluate the similarity between the correlation matrices of the original and synthetic datasets.

        Parameters:
        - method: CorrelationMethod (default: CorrelationMethod.PEARSON); the correlation method to apply.

        Returns:
        - float; the average similarity score between the original and synthetic correlation matrices.
        """
        # Get correlation matrices for the original and synthetic datasets
        orig_corr, syn_corr = self.correlation_pairs(method)
        # Flatten matrices to 1D arrays
        orig_corr_flat = orig_corr.to_numpy().flatten()
        syn_corr_flat = syn_corr.to_numpy().flatten()

        # Calculate similarity score as 1 - mean absolute difference divided by 2
        score = 1 - abs(syn_corr_flat - orig_corr_flat) / 2
        print(f"Method {method} was used.")
        return np.mean(score)  # Return the average score

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics import UtilityMetricCalculator


class JSCalculator(UtilityMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 original_name: str = None,
                 synthetic_name: str = None):
        """
        Initializes the JSCalculator with original and synthetic datasets.

        Parameters:
        - original: pd.DataFrame; the original dataset.
        - synthetic: pd.DataFrame; the synthetic dataset for comparison.
        - original_name: str (default: None); optional name for the original dataset in reporting.
        - synthetic_name: str (default: None); optional name for the synthetic dataset in reporting.
        """
        super().__init__(original, synthetic, original_name=original_name, synthetic_name=synthetic_name)

    def compute_js_similarity(self, bins='auto'):
        """
        Computes the Jensen-Shannon similarity for each feature column between
        the original and synthetic datasets.

        Parameters:
        - bins: int or str (default: 'auto'); number or method for histogram bins.

        Returns:
        - dict; keys are feature names, and values are Jensen-Shannon similarity scores (0 to 1).
        """
        # Retrieve transformed and normalized data from original and synthetic datasets
        original = self.original.transformed_normalized_data
        synthetic = self.synthetic.transformed_normalized_data

        js_similarities = {}
        # Calculate JS similarity for each column in the dataset
        for col in original.columns:
            # Determine range for histogram based on minimum and maximum values in both datasets
            hist_range = (min(original[col].min(), synthetic[col].min()), max(original[col].max(), synthetic[col].max()))

            # Compute histograms for both datasets, normalizing for probability distributions
            orig_hist, bin_edges = np.histogram(original[col], bins=bins, range=hist_range, density=True)
            syn_hist, _ = np.histogram(synthetic[col], bins=bin_edges, density=True)

            # Replace zeros with a small value to avoid division by zero
            orig_hist = np.where(orig_hist == 0, 1e-10, orig_hist)
            syn_hist = np.where(syn_hist == 0, 1e-10, syn_hist)

            # Compute Jensen-Shannon distance and derive similarity (higher is better)
            js_distance = jensenshannon(orig_hist, syn_hist)
            js_similarity = 1 - js_distance  # Convert distance to similarity score
            js_similarities[col] = js_similarity
        return js_similarities  # Return JS similarities for all features

    def evaluate(self):
        """
        Evaluates the overall Jensen-Shannon similarity score across all features.

        Returns:
        - float; average Jensen-Shannon similarity score across features.
        """
        # Compute Jensen-Shannon similarities for each feature
        js_similarities = self.compute_js_similarity()
        # Calculate mean similarity score across all features
        overall_js_similarity_score = np.mean(list(js_similarities.values()))
        return overall_js_similarity_score


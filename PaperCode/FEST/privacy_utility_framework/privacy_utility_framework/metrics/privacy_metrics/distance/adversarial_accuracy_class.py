import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics import PrivacyMetricCalculator


class AdversarialAccuracyCalculator(PrivacyMetricCalculator):
    """Calculate nearest neighbors and adversarial accuracy metrics for original and synthetic datasets."""

    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 distance_metric: str = 'euclidean',
                 original_name: str = None,
                 synthetic_name: str = None):
        """
        Initializes the AdversarialAccuracyCalculator with original and synthetic datasets and a distance metric.
        # reference paper: https://github.com/yknot/ESANN2019/blob/master/metrics/nn_adversarial_accuracy.py

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            distance_metric (str): The metric for calculating distances (default: 'euclidean').
        """
        # Initialize the superclass with datasets and settings
        super().__init__(original, synthetic,
                         distance_metric=distance_metric,
                         original_name=original_name,
                         synthetic_name=synthetic_name)

        # Validate that distance_metric is set
        if distance_metric is None:
            raise ValueError("Parameter 'distance_metric' is required in AdversarialAccuracyCalculator.")

        # Define distance metric
        self.distance_metric = distance_metric

    def evaluate(self):
        """
        Calculate the Nearest Neighbor Adversarial Accuracy (NNAA).

        Returns:
            float: The calculated NNAA.
        """
        # Calculate minimum distances between records in original and synthetic data
        min_d_syn_orig, min_d_orig_syn, min_d_orig_orig, min_d_syn_syn = self._calculate_min_distances()

        # Compute NNAA based on distances within and between datasets
        term1 = np.mean(min_d_orig_syn > min_d_orig_orig)
        term2 = np.mean(min_d_syn_orig > min_d_syn_syn)

        nnaa_value = 0.5 * (term1 + term2)

        return nnaa_value

    def _calculate_min_distances(self):
        """
        Calculate minimum distances for nearest neighbor adversarial accuracy.


        Returns:
        tuple: (min_d_syn_orig, min_d_orig_syn, min_d_orig_orig, min_d_syn_syn)
               - min_d_syn_orig: Minimum distance from each synthetic sample to original samples.
               - min_d_orig_syn: Minimum distance from each original sample to synthetic samples.
               - min_d_orig_orig: Minimum leave-one-out distance within original samples.
               - min_d_syn_syn: Minimum leave-one-out distance within synthetic samples.
        """
        # The transformed and normalized data is used for the NNAA
        original = self.original.transformed_normalized_data
        synthetic = self.synthetic.transformed_normalized_data

        # Calculate distances from synthetic to original
        d_syn_orig = distance.cdist(synthetic, original, metric=self.distance_metric)
        min_d_syn_orig = np.min(d_syn_orig, axis=1)

        # Calculate distances from original to synthetic
        d_orig_syn = distance.cdist(original, synthetic, metric=self.distance_metric)
        min_d_orig_syn = np.min(d_orig_syn, axis=1)

        # Calculate distances within original samples (leave-one-out)
        d_orig_orig = distance.cdist(original, original, metric=self.distance_metric)
        np.fill_diagonal(d_orig_orig, np.inf)  # Ignore self-distances
        min_d_orig_orig = np.min(d_orig_orig, axis=1)

        # Calculate distances within synthetic samples (leave-one-out)
        d_syn_syn = distance.cdist(synthetic, synthetic, metric=self.distance_metric)
        np.fill_diagonal(d_syn_syn, np.inf)  # Ignore self-distances
        min_d_syn_syn = np.min(d_syn_syn, axis=1)

        return min_d_syn_orig, min_d_orig_syn, min_d_orig_orig, min_d_syn_syn


class AdversarialAccuracyCalculator_NN(PrivacyMetricCalculator):
    """Calculate nearest neighbors and adversarial accuracy metrics for original and synthetic datasets using Nearest
    Neighbors (may be faster in some cases)."""

    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 distance_metric: str = 'euclidean',
                 original_name: str = None,
                 synthetic_name: str = None):
        """
        Initialize the AdversarialAccuracyCalculator_NN with datasets and distance metric.

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            distance_metric (str): Metric for distance calculation (default: 'euclidean').
        """
        super().__init__(original, synthetic,
                         distance_metric=distance_metric,
                         original_name=original_name,
                         synthetic_name=synthetic_name)
        if distance_metric is None:
            raise ValueError("Parameter 'distance_metric' is required in AdversarialAccuracyCalculator.")
        # Define distance metric and data for calculation
        self.distance_metric = distance_metric
        self.data = {'original': self.original.transformed_normalized_data,
                     'synthetic': self.synthetic.transformed_normalized_data}
        self.dists = {}

    def _nearest_neighbors(self, t, s):
        """
        Calculate nearest neighbors between two datasets (t and s).

        Parameters:
            t (str): Target dataset name ('original' or 'synthetic').
            s (str): Source dataset name ('original' or 'synthetic').

        Returns:
            tuple: (target dataset, source dataset, distances).
        """
        nn_s = NearestNeighbors(n_neighbors=2, metric=self.distance_metric).fit(self.data[s])
        if t == s:
            # Find distances within the same dataset, taking the second nearest neighbor
            dists, _ = nn_s.kneighbors()
            d = dists[:, 1].reshape(-1, 1)
        else:
            # Find distances between different datasets
            d = nn_s.kneighbors(self.data[t])[0]

        return t, s, d

    def _compute_nn(self):
        """Compute nearest neighbors for all pairs of original and synthetic datasets."""
        pairs = [('original', 'original'), ('original', 'synthetic'), ('synthetic', 'synthetic'),
                 ('synthetic', 'original')]
        for (t, s) in tqdm(pairs):
            t, s, d = self._nearest_neighbors(t, s)
            self.dists[(t, s)] = d

    def _adversarial_accuracy(self):
        """Calculate the adversarial accuracy score based on nearest neighbor distances."""
        orig_vs_synth = np.mean(self.dists[('original', 'synthetic')] > self.dists[('original', 'original')])
        synth_vs_orig = np.mean(self.dists[('synthetic', 'original')] > self.dists[('synthetic', 'synthetic')])
        return 0.5 * (orig_vs_synth + synth_vs_orig)

    def evaluate(self):
        """Run nearest neighbor computation and calculate adversarial accuracy."""
        self._compute_nn()
        return self._adversarial_accuracy()

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance_nd
import ot
from enum import Enum
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics import UtilityMetricCalculator


class WassersteinMethod(Enum):
    # Enumeration for different Wasserstein distance calculation methods
    SINKHORN = "sinkhorn"                       # Sinkhorn Wasserstein distance
    WASSERSTEIN = "wasserstein"                 # Classic Wasserstein distance
    WASSERSTEIN_SAMPLE = "wasserstein_sample"   # Wasserstein distance calculated from sampled data


class WassersteinCalculator(UtilityMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 original_name: str = None,
                 synthetic_name: str = None):
        """
        Initializes the WassersteinCalculator with original and synthetic datasets.

        Parameters:
        - original: pd.DataFrame; the original dataset used for comparison.
        - synthetic: pd.DataFrame; the synthetic dataset generated for analysis.
        - original_name: str (default: None); the name of the original dataset for reporting.
        - synthetic_name: str (default: None); the name of the synthetic dataset for reporting.
        """
        super().__init__(original, synthetic, original_name=original_name, synthetic_name=synthetic_name)

    def evaluate(self, metric=WassersteinMethod.WASSERSTEIN, n_samples=500, n_iterations=1):
        """
        Evaluates the Wasserstein distance between the original and synthetic datasets using the specified method.

        Parameters:
        - metric: WassersteinMethod; the method used to compute the distance (default: WASSERSTEIN).
        - n_samples: int (default: 500); the number of samples to use when sampling the datasets.
        - n_iterations: int (default: 1); the number of iterations for sampling when using the sampled method.

        Returns:
        - The computed Wasserstein distance based on the selected method.
        """
        # Retrieve transformed and normalized data from original and synthetic datasets
        original = self.original.transformed_normalized_data
        synthetic = self.synthetic.transformed_normalized_data
        if metric == WassersteinMethod.SINKHORN:
            # Parameters for Sinkhorn distance calculation
            numItermax = 1000  # Maximum number of iterations for convergence
            stopThr = 1e-9  # Stopping threshold for convergence
            reg = 0.0025  # Regularization term for Sinkhorn distance

            # Convert data to numpy arrays for distance calculation
            original = original.to_numpy()
            synthetic = synthetic.to_numpy()

            # Compute pairwise distance matrix between original and synthetic datasets
            M = ot.dist(original, synthetic, metric="euclidean")

            # Compute Sinkhorn approximation of the Wasserstein distance
            wass_dist = ot.sinkhorn2(np.ones((original.shape[0],)) / original.shape[0],
                                     np.ones((synthetic.shape[0],)) / synthetic.shape[0],
                                     M, reg, stopThr=stopThr, numItermax=numItermax)
            print(f'Sinkhorn Wasserstein Distance: {wass_dist}')
            return wass_dist
        elif metric == WassersteinMethod.WASSERSTEIN:
            # Compute classic Wasserstein distance
            distance = wasserstein_distance_nd(original, synthetic)
            print(f"Wasserstein Distance: {distance}")
            return distance
        if metric == WassersteinMethod.WASSERSTEIN_SAMPLE:
            # List to store distances from each sampled iteration
            distances = []

            # Loop over the number of iterations for sampling
            for _ in range(n_iterations):
                # Randomly sample a subset of the original and synthetic data
                orig_sample = original.sample(n=n_samples, random_state=np.random.randint(0, 10000))
                syn_sample = synthetic.sample(n=n_samples, random_state=np.random.randint(0, 10000))

                # Compute the Wasserstein distance for the sampled data
                dist = wasserstein_distance_nd(orig_sample, syn_sample)
                distances.append(dist)  # Store the computed distance in the list

            # Calculate the mean distance from all iterations
            sampled_dist = np.mean(distances)
            print(f"Sampled Wasserstein Distance: {sampled_dist}, with n_samples={n_samples}, n_iterations={n_iterations}")

            # Return the average of all computed distances
            return sampled_dist

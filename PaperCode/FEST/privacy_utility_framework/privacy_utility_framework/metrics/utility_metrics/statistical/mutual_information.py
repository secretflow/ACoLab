import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import normalized_mutual_info_score

from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics import UtilityMetricCalculator


class MICalculator(UtilityMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 original_name: str = None,
                 synthetic_name: str = None):
        """
        Initializes the MICalculator with original and synthetic datasets.

        Parameters:
        - original: pd.DataFrame; the original dataset.
        - synthetic: pd.DataFrame; the synthetic dataset generated for analysis.
        - original_name: str (default: None); the name of the original dataset for reporting.
        - synthetic_name: str (default: None); the name of the synthetic dataset for reporting.
        """
        super().__init__(original, synthetic, original_name=original_name, synthetic_name=synthetic_name)

    @staticmethod
    def pairwise_attributes_mutual_information(dataset: pd.DataFrame):
        """
        Computes normalized mutual information for all attribute pairs in a dataset.

        Parameters:
        - dataset: pd.DataFrame; the dataset to compute mutual information on.

        Returns:
        - pd.DataFrame; a DataFrame containing mutual information values for each attribute pair.
        """
        sorted_columns = sorted(dataset.columns)  # Sort columns for consistent ordering
        mi_df = DataFrame(columns=sorted_columns, index=sorted_columns, dtype=float)

        # Compute mutual information for each attribute pair
        for row in mi_df.columns:
            for col in mi_df.columns:
                mi_df.loc[row, col] = normalized_mutual_info_score(dataset[row].astype(str),
                                                                   dataset[col].astype(str),
                                                                   average_method='arithmetic')
        return mi_df

    def evaluate(self):
        """
         Evaluates the similarity in mutual information (MI) structure between original and synthetic datasets.

         Returns:
         - float; the average similarity score based on mutual information between original and synthetic datasets.
         """
        # Retrieve transformed and normalized data from original and synthetic datasets
        original = self.original.transformed_normalized_data
        synthetic = self.synthetic.transformed_normalized_data

        # Calculate pairwise mutual information for the original and synthetic datasets
        private_mi = self.pairwise_attributes_mutual_information(original)
        synthetic_mi = self.pairwise_attributes_mutual_information(synthetic)

        # Flatten mutual information DataFrames for direct comparison
        private_mi_flat = private_mi.to_numpy().flatten()
        synthetic_mi_flat = synthetic_mi.to_numpy().flatten()

        # Calculate the similarity score between the original and synthetic datasets
        score = 1 - abs(synthetic_mi_flat - private_mi_flat) / 2
        return np.mean(score)  # Return the mean similarity score as the final evaluation result

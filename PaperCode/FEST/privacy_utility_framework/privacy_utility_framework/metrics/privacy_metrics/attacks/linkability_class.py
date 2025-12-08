from typing import List, Tuple, Optional
import pandas as pd
from anonymeter.evaluators import LinkabilityEvaluator

from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics import PrivacyMetricCalculator


class LinkabilityCalculator(PrivacyMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 aux_cols: Tuple[List[str], List[str]],
                 n_attacks: Optional[int] = 500,
                 n_neighbors: int = 1,
                 control: Optional[pd.DataFrame] = None,
                 original_name: str = None, synthetic_name: str = None,):
        """
        Initializes the LinkabilityCalculator instance for evaluating linkability risks.

        Parameters:
        - original (pd.DataFrame): The original dataset.
        - synthetic (pd.DataFrame): The synthetic dataset generated from the original data.
        - aux_cols (Tuple[List[str], List[str]]): A tuple containing two lists of auxiliary columns used for linkability assessment.
        - n_attacks (Optional[int]): The number of attacks to perform. Defaults to 500.
        - n_neighbors (int): The number of neighbors to consider for linkability. Defaults to 1.
        - control (Optional[pd.DataFrame]): An optional control dataset for evaluating linkability risk.
        - original_name (str, optional): An optional name for the original dataset.
        - synthetic_name (str, optional): An optional name for the synthetic dataset.

        Raises:
        ValueError: If the aux_cols parameter is not provided.
        """
        super().__init__(original, synthetic, aux_cols=aux_cols,
                         original_name=original_name, synthetic_name=synthetic_name)
        if aux_cols is None:
            raise ValueError("Parameter 'aux_cols' is required in LinkabilityCalculator.")
        self.aux_cols = aux_cols
        self.n_attacks = min(n_attacks, len(control))
        self.n_neighbors = n_neighbors
        self.control = control

    def evaluate(self):
        """
        Evaluates the linkability risk between the original and synthetic datasets.

        Returns:
        The risk assessment result from the LinkabilityEvaluator.
        """
        # Retrieve the data from the original and synthetic Dataset objects (no need for normalization or
        # transformation)
        original = self.original.data
        synthetic = self.synthetic.data
        evaluator = LinkabilityEvaluator(
            ori=original,
            syn=synthetic,
            aux_cols=self.aux_cols,
            n_attacks=self.n_attacks,
            n_neighbors=self.n_neighbors,
            control=self.control
        )
        return evaluator.evaluate().risk()

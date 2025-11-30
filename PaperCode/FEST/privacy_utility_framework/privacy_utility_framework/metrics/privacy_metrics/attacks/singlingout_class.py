from typing import Optional
import pandas as pd
from anonymeter.evaluators import SinglingOutEvaluator

from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics import PrivacyMetricCalculator


class SinglingOutCalculator(PrivacyMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 n_cols: int = 3,
                 max_attempts: Optional[int] = 10000000,
                 n_attacks: int = 500,
                 control: Optional[pd.DataFrame] = None,
                 original_name: str = None, synthetic_name: str = None):
        """
        Initializes the SinglingOutCalculator.

        Parameters:
        - original: pd.DataFrame; the original dataset used for comparison.
        - synthetic: pd.DataFrame; the synthetic dataset generated for analysis.
        - n_cols: int (default: 3); the number of columns to evaluate for the singling out risk.
        - max_attempts: Optional[int] (default: 10000000); the maximum number of attempts during evaluation.
        - n_attacks: int (default: 500); the number of attack simulations to conduct.
        - control: Optional[pd.DataFrame] (default: None); a control dataset used to evaluate the attack effectiveness.
        - original_name (str, optional): An optional name for the original dataset.
        - synthetic_name (str, optional): An optional name for the synthetic dataset.
        """
        super().__init__(original, synthetic, original_name=original_name, synthetic_name=synthetic_name)
        self.n_cols = n_cols
        self.n_attacks = n_attacks
        self.control = control
        self.max_attempts = max_attempts

    def evaluate(self):
        """
        Evaluates the risk of singling out an individual in the synthetic dataset.

        Returns:
        - The calculated risk associated with the singling out attack.
        """
        # Retrieve the data from the original and synthetic Dataset objects (no need for normalization or
        # transformation)
        original = self.original.data
        synthetic = self.synthetic.data

        evaluator = SinglingOutEvaluator(
            ori=original,
            syn=synthetic,
            n_attacks=self.n_attacks,
            n_cols=self.n_cols,
            control=self.control,
            max_attempts=self.max_attempts,
        )
        return evaluator.evaluate().risk()
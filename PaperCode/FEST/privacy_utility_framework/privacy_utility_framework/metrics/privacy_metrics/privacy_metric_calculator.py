from abc import ABC, abstractmethod
import pandas as pd

from privacy_utility_framework.privacy_utility_framework.dataset.dataset import DatasetManager


class PrivacyMetricCalculator(ABC):
    """
    Abstract base class for privacy metric calculators, providing data validation
    and transformation methods for original and synthetic datasets.

    Parameters
    ----------
    original : pd.DataFrame
        The original dataset to compare against the synthetic data.
    synthetic : pd.DataFrame
        The synthetic dataset generated to resemble the original data.
    original_name : str, optional
        Name for the original dataset.
    synthetic_name : str, optional
        Name for the synthetic dataset.
    """

    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame, original_name: str = None,
                 synthetic_name: str = None, **kwargs):
        # Initialize attributes for original and synthetic data
        self.synthetic = None
        self.original = None

        # Ensure both original and synthetic datasets are pandas DataFrames
        if not isinstance(original, pd.DataFrame):
            raise TypeError("original_data must be an instance of pandas Dataframe.")
        if not isinstance(synthetic, pd.DataFrame):
            raise TypeError("synthetic_data must be an instance of pandas Dataframe.")
        # Perform data transformation and normalization
        self._transform_and_normalize(original, synthetic, original_name, synthetic_name)
        # Perform data validation to ensure compatibility between datasets
        self._validate_data()

    @abstractmethod
    def evaluate(self) -> float:
        """
        Abstract method for metric evaluation. Must be implemented in subclasses.

        Returns
        -------
        float
            The calculated privacy metric score.
        """
        pass

    def _validate_data(self):
        """
       Validates that the original and synthetic datasets have the same structure,
       no missing values, and compatible data types.

       Raises
       ------
       ValueError
           If the datasets do not match in structure or have incompatible data types.
        """
        # Check that column names match between original and synthetic datasets
        if set(self.original.data.columns) != set(self.synthetic.data.columns):
            raise ValueError("Column names do not match between original and synthetic datasets.")

        # Check that the number of columns matches
        if len(self.original.data.columns) != len(self.synthetic.data.columns):
            raise ValueError("Number of columns do not match between original and synthetic datasets.")

        # Ensure no missing values in either dataset
        assert not self.original.data.isnull().any().any(), "Original dataset contains missing values."
        assert not self.synthetic.data.isnull().any().any(), "Synthetic dataset contains missing values."

        # Confirm data types are consistent across columns in both datasets
        for col in self.original.data.columns:
            if self.original.data[col].dtype != self.synthetic.data[col].dtype:
                raise ValueError(f"Data type mismatch in column '{col}'.")

    def _transform_and_normalize(self, original, synthetic, original_name, synthetic_name):
        """
        Transforms and normalizes both the original and synthetic datasets.

        Parameters
        ----------
        original : pd.DataFrame
            The original dataset to transform and normalize.
        synthetic : pd.DataFrame
            The synthetic dataset to transform and normalize.
        original_name : str
            The name of the original dataset.
        synthetic_name : str
            The name of the synthetic dataset.
        """
        # Initialize DatasetManager with original and synthetic data and their names
        manager = DatasetManager(original, synthetic, original_name, synthetic_name)

        # Configure the transformer and scaler to apply transformations to datasets
        manager.set_transformer_and_scaler_for_datasets()

        # Perform transformation and normalization on both datasets
        manager.transform_and_normalize_datasets()

        # Store transformed and normalized datasets in attributes
        self.original = manager.original_dataset
        self.synthetic = manager.synthetic_dataset

import pandas as pd
from rdt import HyperTransformer
from rdt.transformers import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    def __init__(self, data: pd.DataFrame, name=""):
        """
        Initialize a dataset with data and an optional name.

        Args:
            data (pd.DataFrame): The dataset to be managed.
            name (str): Optional name for the dataset.
        """
        self.data = data
        self.name = name
        self.scaler = None
        self.transformer = None
        self.transformed_data = None
        self.transformed_normalized_data = None

    def set_transformer_and_scaler(self):
        """
        Sets up and fits the transformer and scaler for the dataset, if not already set.
        Configures a transformer for categorical encoding and a MinMaxScaler for normalization.
        """
        # Initialize transformer and configure it for categorical columns
        self.transformer = HyperTransformer()
        self.transformer.detect_initial_config(data=self.data)
        config = self.transformer.get_config()

        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        config['transformers'].update({col: OneHotEncoder() for col in categorical_columns})

        # Fit transformer to the original dataset
        self.transformer.fit(self.data)

        # Initialize and fit the MinMaxScaler for numeric columns in transformed data
        numeric_cols = self.transformer.transform(self.data).select_dtypes(include=[float, int]).columns
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.transformer.transform(self.data)[numeric_cols])
        #print(f"Transformer and scaler have been set and fitted for {self.name}.")

    def transform(self):
        """
        Applies the transformation using the fitted transformer.
        Raises an error if the transformer is not set.
        """
        if self.transformer is None:
            raise RuntimeError("Transformer must be set before transformation.")
        self.transformed_data = self.transformer.transform(self.data)
        #print(f"Data transformed for {self.name}.")

    def normalize(self):
        """
        Applies normalization using the scaler on the transformed data.
        Raises an error if the data has not been transformed.
        """
        if self.transformed_data is None:
            raise RuntimeError("Data must be transformed before normalization.")

        numeric_cols = self.transformed_data.select_dtypes(include=[float, int]).columns
        self.transformed_normalized_data = self.transformed_data.copy()
        self.transformed_normalized_data[numeric_cols] = self.scaler.transform(self.transformed_data[numeric_cols])
        #print(f"Data normalized for {self.name}.")

    def set_transformer_and_scaler_from(self, other_dataset):
        """
        Sets the transformer and scaler from another dataset to ensure consistent transformation and scaling.

        Args:
            other_dataset (Dataset): The dataset from which to copy the transformer and scaler.
        """
        self.transformer = other_dataset.transformer
        self.scaler = other_dataset.scaler


class DatasetManager:
    def __init__(self, original, synthetic, original_name=None, synthetic_name=None):
        """
        Initialize with original and synthetic Dataset objects.

        Args:
            original (pd.DataFrame): The original dataset.
            synthetic (pd.DataFrame): The synthetic dataset.
        """
        original_name = original_name if original_name is not None else "Original_Dataset"
        synthetic_name = synthetic_name if synthetic_name is not None else "Synthetic_Dataset"
        self.original_dataset = Dataset(original, name=original_name)
        self.synthetic_dataset = Dataset(synthetic, name=synthetic_name)

    def set_transformer_and_scaler_for_datasets(self):
        """
        Sets up and fits the transformer and scaler on the original dataset,
        then applies the same transformer and scaler to the synthetic dataset.
        """
        # Set and fit transformer and scaler for the original dataset
        self.original_dataset.set_transformer_and_scaler()

        # Copy the fitted transformer and scaler to the synthetic dataset
        self.synthetic_dataset.set_transformer_and_scaler_from(self.original_dataset)

    def transform_and_normalize_datasets(self):
        """
        Transforms and normalizes both datasets using the same transformer and scaler.
        Ensures consistent transformation and normalization between both datasets.
        """
        # Ensure that transformer and scaler are set up and copied over
        self.set_transformer_and_scaler_for_datasets()

        # Transform and normalize both datasets using their methods
        self.original_dataset.transform()
        self.original_dataset.normalize()
        self.synthetic_dataset.transform()
        self.synthetic_dataset.normalize()

        #print("Both datasets have been transformed and normalized.")
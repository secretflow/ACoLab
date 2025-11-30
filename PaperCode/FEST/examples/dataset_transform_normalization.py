import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from privacy_utility_framework.privacy_utility_framework.dataset.dataset import DatasetManager


def dataset_example():
    # Sample original and synthetic data
    original_data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': ['cat', 'dog', 'mouse']
    })

    synthetic_data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': ['cat', 'dog', 'mouse']
    })
    # Initialize the DatasetManager
    manager = DatasetManager(original_data, synthetic_data)

    # Set the transformer and scaler for the datasets
    manager.set_transformer_and_scaler_for_datasets()

    # Transform and normalize the datasets
    manager.transform_and_normalize_datasets()

    # Access transformed and normalized data
    print("Original Transformed Data:\n", manager.original_dataset.transformed_data)
    print("Original Normalized Data:\n", manager.original_dataset.transformed_normalized_data)
    print("Synthetic Transformed Data:\n", manager.synthetic_dataset.transformed_data)
    print("Synthetic Normalized Data:\n", manager.synthetic_dataset.transformed_normalized_data)

dataset_example()
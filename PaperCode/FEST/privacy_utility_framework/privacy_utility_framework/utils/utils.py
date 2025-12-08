import pandas as pd
from sklearn.model_selection import train_test_split


def dynamic_train_test_split(df: pd.DataFrame, small_threshold: int = 1000, large_threshold: int = 50000):
    """
    Split the DataFrame into training and test sets with a dynamic test size based on the DataFrame size.

    Parameters:
    - df: pd.DataFrame
      The DataFrame to split.
    - small_threshold: int (default: 1000)
      The threshold below which a smaller test size is used.
    - large_threshold: int (default: 50000)
      The threshold above which a larger test size is used.

    Returns:
    - train_df: pd.DataFrame
      The training set.
    - test_df: pd.DataFrame
      The test set.
    """
    num_rows = len(df)

    if num_rows <= small_threshold:
        test_size = 0.1  # 10% for very small datasets
    elif num_rows <= large_threshold:
        test_size = 0.2  # 20% for medium-sized datasets
    else:
        test_size = 0.3  # 30% for large datasets
    print(f"Test size was: {test_size}")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df

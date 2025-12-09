import numpy as np
import pandas as pd
from dython.nominal import associations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (
    BayesianRidge,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import f1_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class TabulaEvaluator:
    def __init__(
        self, real_data, synthetic_data, target_column, task_type="classification"
    ):
        """
        Initializes the evaluator.

        Args:
            real_data (pd.DataFrame): The original dataset.
            synthetic_data (pd.DataFrame): The generated dataset.
            target_column (str): The name of the target variable column.
            task_type (str): 'classification' or 'regression'.
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.target_col = target_column
        self.task_type = task_type

        # Preprocessing: Encode categorical columns for ML models if necessary
        # The paper implies standard ML evaluation which requires numerical input
        self._preprocess_data()

    def _preprocess_data(self):
        """Simple label encoding for ML utility compatibility."""
        le = LabelEncoder()
        for col in self.real_data.columns:
            if (
                self.real_data[col].dtype == "object"
                or self.real_data[col].dtype.name == "category"
            ):
                # Fit on combined unique values to handle unseen labels in synth
                combined = pd.concat(
                    [self.real_data[col], self.synthetic_data[col]]
                ).unique()
                le.fit(combined)
                self.real_data[col] = le.transform(self.real_data[col])
                self.synthetic_data[col] = le.transform(self.synthetic_data[col])

    def evaluate_ml_utility(self):
        """
        1. Machine Learning Utility
        Splits original data 80/20. Trains on Synthetic, Tests on Real (TSTR).
        """
        print(f"--- Evaluating Machine Learning Utility ({self.task_type}) ---")

        # Split original data into Test set (20%)
        # We use Real Train only if we want to compare Real vs Real,
        # but for Synthetic Utility, we Train on Synthetic and Test on Real.
        _, X_test_real, _, y_test_real = train_test_split(
            self.real_data.drop(columns=[self.target_col]),
            self.real_data[self.target_col],
            test_size=0.2,
            random_state=42,
        )

        # Train set is the full Synthetic dataset
        X_train_syn = self.synthetic_data.drop(columns=[self.target_col])
        y_train_syn = self.synthetic_data[self.target_col]

        results = []

        if self.task_type == "classification":
            # Classification Models
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "SVM": SVC(),
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "MLP": MLPClassifier(max_iter=1000),
            }
            metric_name = "F1-Score"

        elif self.task_type == "regression":
            # Regression Models
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Bayesian Ridge": BayesianRidge(),
            }
            metric_name = "MAPE"

        else:
            raise ValueError("Task type must be 'classification' or 'regression'")

        for name, model in models.items():
            try:
                # Train on Synthetic
                model.fit(X_train_syn, y_train_syn)
                # Test on Real
                predictions = model.predict(X_test_real)

                if self.task_type == "classification":
                    # Metric: F1-score
                    score = f1_score(y_test_real, predictions, average="macro")
                else:
                    # Metric: MAPE
                    score = mean_absolute_percentage_error(y_test_real, predictions)

                results.append(score)
                print(f"{name}: {score:.4f}")
            except Exception as e:
                print(f"Failed to train {name}: {e}")

        # Report average score
        avg_score = np.mean(results)
        print(f"\nAverage {metric_name}: {avg_score:.4f}")
        return avg_score

    def evaluate_statistical_similarity(self):
        """
        2. Statistical Similarity
        Calculates pair-wise correlation matrices for real and synthetic data.
        Uses dython for Pearson (continuous), Uncertainty (categorical),
        and Correlation Ratio (mixed).
        """
        print("\n--- Evaluating Statistical Similarity ---")

        # Calculate correlation matrix for Real Data
        # dython.nominal.associations automatically selects Pearson, Cramer's V/Uncertainty,
        # or Correlation Ratio based on data types.
        real_corr = associations(
            self.real_data, nom_nom_assoc="uncertainty_coefficient", compute_only=True
        )["corr"]

        # Calculate correlation matrix for Synthetic Data
        syn_corr = associations(
            self.synthetic_data,
            nom_nom_assoc="uncertainty_coefficient",
            compute_only=True,
        )["corr"]

        # Fill NaNs (common in correlation calculation) with 0 for distance calculation
        real_corr = real_corr.fillna(0)
        syn_corr = syn_corr.fillna(0)

        # Calculate Correlation Distance (Euclidean/Frobenius norm of the difference)
        # Note: Paper says "lower values indicate higher synthesis quality"
        diff_matrix = real_corr - syn_corr
        correlation_distance = np.linalg.norm(diff_matrix)

        print(f"Correlation Distance: {correlation_distance:.4f}")
        return correlation_distance


# --- Usage Example ---
if __name__ == "__main__":

    real_data = pd.read_csv("./Real_Datasets/Insurance/insurance.csv")
    synthetic_data = pd.read_csv("insurance_400epoch(1).csv")

    evaluator = TabulaEvaluator(
        real_data=real_data,
        synthetic_data=synthetic_data,
        target_column="charges",
        task_type="regression",
    )

    #  Run Evaluations
    ml_utility_score = evaluator.evaluate_ml_utility()
    stat_similarity_score = evaluator.evaluate_statistical_similarity()

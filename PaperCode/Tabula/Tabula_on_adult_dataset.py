import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabula import Tabula

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Preprocess the Adult dataset
def preprocess_adult_data():
    # Define column names for the Adult dataset
    adult_columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]

    # Load training data
    train_data = pd.read_csv(
        "Real_Datasets/Adult/adult.data",
        names=adult_columns,
        sep=", ",
        engine="python",
        na_values="?",
    )

    # Load test data (skip header row)
    test_data = pd.read_csv(
        "Real_Datasets/Adult/adult.test",
        names=adult_columns,
        sep=", ",
        engine="python",
        na_values="?",
        skiprows=1,
    )

    # Combine training and test data
    adult_data = pd.concat([train_data, test_data], ignore_index=True)

    # Remove rows with missing values
    adult_data = adult_data.dropna()

    # Clean 'income' column by removing dots
    adult_data["income"] = adult_data["income"].str.replace(".", "", regex=False)

    # Save preprocessed data
    adult_data.to_csv("Real_Datasets/Adult_compressed.csv", index=False)
    print(
        f"Preprocessing completed! The Adult dataset has {len(adult_data)} rows. Saved to: Real_Datasets/Adult_compressed.csv"
    )


# Execute preprocessing
preprocess_adult_data()

# Load preprocessed data
data = pd.read_csv("Real_Datasets/Adult_compressed.csv")


# Specify categorical columns for the Adult dataset
categorical_columns = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "income",
]

# Initialize Tabula model
model = Tabula(
    llm="distilgpt2",
    experiment_dir="adult_training",
    batch_size=32,
    epochs=50,
    categorical_columns=categorical_columns,
)


# Load pretrained model weights
model.model.load_state_dict(
    torch.load("pretrained-model/tabula_pretrained_model.pt"), strict=False
)


# Train the model
model.fit(data.sample(n=500))


# Save trained model
torch.save(model.model.state_dict(), "adult_training/model_50epoch.pt")

# Generate synthetic data (n_samples close to original data size)
synthetic_data = model.sample(n_samples=45000, max_length=150)

# Save synthetic data
synthetic_data.to_csv("adult_50epoch.csv", index=False)
print("Synthetic data generation completed! Saved to: adult_50epoch.csv")


# Evaluate synthetic data utility
def encode_data(df):
    # Encode categorical columns
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
    return df


# Encode real and synthetic data
real_data_encoded = encode_data(data.copy())
synth_data_encoded = encode_data(synthetic_data.copy())

# Split features and target (target: 'income')
X_real, y_real = real_data_encoded.drop("income", axis=1), real_data_encoded["income"]
X_synth, y_synth = (
    synth_data_encoded.drop("income", axis=1),
    synth_data_encoded["income"],
)

# Split real data into train and test sets
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42
)

# Train Random Forest for evaluation
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Evaluate with real training data
rf.fit(X_real_train, y_real_train)
y_pred_real = rf.predict(X_real_test)
real_f1 = f1_score(y_real_test, y_pred_real, average="weighted")

# Evaluate with synthetic data
rf.fit(X_synth, y_synth)
y_pred_synth = rf.predict(X_real_test)
synth_f1 = f1_score(y_real_test, y_pred_synth, average="weighted")

# Output evaluation results
print(f"Real data F1-score: {real_f1:.4f}")
print(f"Synthetic data F1-score: {synth_f1:.4f}")

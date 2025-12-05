import os

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabula import Tabula

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Preprocess loan data
# Read Excel file and locate 'Data' sheet
excel_file = pd.ExcelFile("Real_Datasets/Loan/Bank_Personal_Loan_Modelling.xlsx")
data = excel_file.parse("Data")

# Process column names (replace spaces with underscores)
data.columns = data.columns.str.replace(" ", "_")

# Define categorical columns and set their type
categorical_columns = [
    "Family",
    "Education",
    "Securities_Account",
    "CD_Account",
    "Online",
    "CreditCard",
    "Personal_Loan",
]
for col in categorical_columns:
    data[col] = data[col].astype("object")

# Split into train and test sets (4k train, 1k test) with stratified sampling
train_data, test_data = train_test_split(
    data,
    train_size=4000,
    test_size=1000,
    random_state=42,
    stratify=data["Personal_Loan"],
)
loan_full = pd.concat([train_data, test_data], ignore_index=True)

# Save preprocessed data
loan_full.to_csv("Real_Datasets/Loan_compressed.csv", index=False)
print("Preprocessing completed. Saved to: Real_Datasets/Loan_compressed.csv")

# Load preprocessed data
data = pd.read_csv("Real_Datasets/Loan_compressed.csv")
print("Data shape:", data.shape)
print("Categorical columns:", data.select_dtypes(include=["object"]).columns.tolist())

# Initialize Tabula model
model = Tabula(
    llm="distilgpt2",
    experiment_dir="loan_training",
    batch_size=8,
    epochs=100,
    categorical_columns=categorical_columns,
)

# Load pretrained model (uncomment to use randomly initialized model)
model.model.load_state_dict(
    torch.load("pretrained-model/tabula_pretrained_model.pt"), strict=False
)

# Train the model
model.fit(data)

# Save trained model
torch.save(model.model.state_dict(), "loan_training/model_100epoch.pt")

# Generate synthetic data
synthetic_data = model.sample(n_samples=5000, max_length=100)
synthetic_data.to_csv("loan_100epoch.csv", index=False)


# Evaluate F1-score
def encode_data(df):
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
    return df


real_encoded = encode_data(data.copy())
synth_encoded = encode_data(synthetic_data.copy())

X_real, y_real = (
    real_encoded.drop("Personal_Loan", axis=1),
    real_encoded["Personal_Loan"],
)
X_synth, y_synth = (
    synth_encoded.drop("Personal_Loan", axis=1),
    synth_encoded["Personal_Loan"],
)

# Train with synthetic data, evaluate with real test set
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_synth, y_synth)
y_pred = rf.predict(X_real.sample(frac=0.2, random_state=42))
synth_f1 = f1_score(
    y_real.sample(frac=0.2, random_state=42), y_pred, average="weighted"
)
print(f"Synthetic data F1-score: {synth_f1:.4f}")

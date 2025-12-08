import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch

from tabula import Tabula

data = pd.read_csv("Real_Datasets/King_compressed.csv")
print("King dataset shape:", data.shape)

# Define categorical columns for King dataset
categorical_columns = [
    "bedrooms",
    "bathrooms",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "zipcode",
    "price",
]

# Convert categorical columns to object type for model recognition
for col in categorical_columns:
    if col in data.columns:
        data[col] = data[col].astype("object")
        print(f"✅ Column {col} converted to categorical type")
    else:
        print(f"⚠️  Warning: Column {col} does not exist")

# Verify categorical columns
cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
print("\nKing dataset final number of categorical columns:", len(cat_cols))
print("Categorical columns list:", cat_cols)

print("Data shape:", data.shape)
print("Categorical columns:", data.select_dtypes(include=["object"]).columns.tolist())


model = Tabula(
    llm="distilgpt2",
    experiment_dir="king_training",
    batch_size=8, 
    epochs=300,  
    categorical_columns=categorical_columns,
)


model.model.load_state_dict(
    torch.load("pretrained-model/tabula_pretrained_model.pt"), strict=False
)

model.fit(data)

torch.save(model.model.state_dict(), "king_training/model_300epoch.pt")
print(
    "King dataset training completed, model saved to: king_training/model_300epoch.pt"
)

# Generate synthetic data with similar volume to original
synthetic_data = model.sample(
    n_samples=21613, max_length=250  # Match training max_length to avoid truncation
)

synthetic_data.to_csv("king_300epoch.csv", index=False)
print("Synthetic data generated, saved to: king_300epoch.csv")
print("Synthetic data shape:", synthetic_data.shape)

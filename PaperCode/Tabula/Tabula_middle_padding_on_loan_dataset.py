import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch

from tabula_middle_padding import Tabula

data = pd.read_csv("./Real_Datasets/Loan_compressed.csv")

# categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
model = Tabula(
    llm="distilgpt2", experiment_dir="loan_training", batch_size=8, epochs=100
)

model.fit(data, conditional_col=data.columns[0])

torch.save(model.model.state_dict(), "loan_training/model_100epoch(1).pt")

synthetic_data = model.sample(n_samples=5000, max_length=100)
synthetic_data.to_csv("loan_100epoch(1).csv", index=False)

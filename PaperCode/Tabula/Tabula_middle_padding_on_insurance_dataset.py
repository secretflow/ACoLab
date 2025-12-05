import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch

from tabula_middle_padding import Tabula

data = pd.read_csv("./Real_Datasets/Insurance/insurance.csv")

# categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
model = Tabula(
    llm="distilgpt2", experiment_dir="insurance_training", batch_size=8, epochs=400
)

model.fit(data, conditional_col=data.columns[0])

torch.save(model.model.state_dict(), "insurance_training/model_400epoch(1).pt")

synthetic_data = model.sample(n_samples=1338)
synthetic_data.to_csv("insurance_400epoch(1).csv", index=False)

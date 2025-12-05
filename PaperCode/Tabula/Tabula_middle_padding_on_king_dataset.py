import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch

from tabula_middle_padding import Tabula

data = pd.read_csv("./Real_Datasets/King_compressed.csv")

# categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
model = Tabula(
    llm="distilgpt2", experiment_dir="king_training", batch_size=32, epochs=300
)

model.fit(data, conditional_col=data.columns[0])

torch.save(model.model.state_dict(), "king_training/model_300epoch(1).pt")

synthetic_data = model.sample(n_samples=21613, max_length=250)
synthetic_data.to_csv("king_300epoch.csv(1)", index=False)

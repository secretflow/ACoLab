import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch

from tabula_middle_padding import Tabula

data = pd.read_csv("./Real_Datasets/Adult_compressed.csv")

# categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
model = Tabula(
    llm="distilgpt2", experiment_dir="adult_training", batch_size=32, epochs=50
)

model.fit(data, conditional_col=data.columns[0])

torch.save(model.model.state_dict(), "adult_training/model_50epoch(1).pt")

synthetic_data = model.sample(n_samples=6000)
synthetic_data.to_csv("adult_50epoch.csv(1)", index=False)

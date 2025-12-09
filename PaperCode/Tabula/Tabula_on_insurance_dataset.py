import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd

from tabula import Tabula

data = pd.read_csv("Real_Datasets/Insurance/insurance.csv")


categorical_columns = ["sex", "children", "sm", "region"]
model = Tabula(
    llm="distilgpt2",
    experiment_dir="insurance_training",
    batch_size=32,
    epochs=400,
    categorical_columns=categorical_columns,
)


# Comment this block out to test tabula starting from randomly initialized model.
# Comment this block out when uses tabula_middle_padding
import torch

model.model.load_state_dict(
    torch.load("pretrained-model/tabula_pretrained_model.pt"), strict=False
)

model.fit(data)




torch.save(model.model.state_dict(), "insurance_training/model_400epoch.pt")


synthetic_data = model.sample(n_samples=1338)
synthetic_data.to_csv("insurance_400epoch.csv", index=False)

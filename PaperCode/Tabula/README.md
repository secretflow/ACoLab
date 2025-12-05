# TabuLa: Harnessing Language Models for Tabular Data Synthesis

This repo is a reproduction of the paper: [TabuLa: Harnessing Language Models for Tabular Data Synthesis](https://arxiv.org/abs/2310.12746)  
This repository is developed based on an existing open-source repository: [Tabula](https://github.com/zhao-zilong/Tabula)


## I. Environment Preparation

### 1.1 Creating a Virtual Environment
```bash
# Create a conda environment
conda create -n tabula python=3.10.19
conda activate tabula
```

### 1.2 Installing Dependencies
```bash
# Install basic dependencies
pip install -r requirements.txt
```


## II. Data and Model Preparation

### 2.1 Configuring the Kaggle Environment (for subsequent dataset downloads)
- **Step 1: Register for a Kaggle account**  
  Visit the link: https://www.kaggle.com/ and complete the account registration.  

- **Step 2: Obtain a Kaggle API key**  
  After logging in, go to account settings and download the API key file (named `kaggle.json`).  

- **Step 3: Upload the key to the specified directory**  
  Use Cursor's "File Upload" function to upload `kaggle.json` to the `~/Tabula` directory.  

- **Step 4: Execute configuration commands**  
  Run the following commands to complete Kaggle configuration:  
  ```bash
  # Create a Kaggle configuration directory
  mkdir -p ~/.kaggle
  # Move the API key to the configuration directory
  mv ~/Tabula/kaggle.json ~/.kaggle/
  # Set permissions (to avoid permission errors)
  chmod 600 ~/.kaggle/kaggle.json
  ```  


### 2.2 Downloading the Pretrained Model
- **Objective**: Download the model to the `./pretrained-model` directory  
- **Command to execute**:  
  ```bash
  # Create a directory for storing the model
  mkdir -p ./pretrained-model
  # Download the model using gdown (the --fuzzy parameter is used to identify Google Drive share links)
  gdown https://drive.google.com/file/d/1_YxelekxY5MXhgn93MYgsZEEfBYAy7h6/view?usp=sharing --fuzzy -O ./pretrained-model/tabula_pretrained_model.pt
  ```  


### 2.3 Downloading Various Datasets

#### 2.3.1 Insurance Dataset
- **Objective**: Download and store in the `./Real_Datasets/Insurance` directory  
- **Command to execute**:  
  ```bash
  # Create a dataset directory
  mkdir -p ./Real_Datasets/Insurance
  # Download and unzip the dataset to the specified directory
  kaggle datasets download -d mirichoi0218/insurance --unzip -p ./Real_Datasets/Insurance
  ```  


#### 2.3.2 Adult Dataset
- **Objective**: Download and store in the `./Real_Datasets/Adult` directory  
- **Command to execute**:  
  ```bash
  # Create a dataset directory
  mkdir -p ./Real_Datasets/Adult
  # Download the data file (adult.data)
  wget -P ./Real_Datasets/Adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
  ```  


#### 2.3.3 Loan Dataset
- **Objective**: Download and store in the `./Real_Datasets/Loan` directory  
- **Command to execute**:  
  ```bash
  # Create a dataset directory
  mkdir -p ./Real_Datasets/Loan
  # Download and unzip the dataset to the specified directory
  kaggle datasets download -d itsmesunil/bank-loan-modelling --unzip -p ./Real_Datasets/Loan
  ```  


#### 2.3.4 King Dataset 
- **Objective**: Download and store in the `./Real_Datasets` directory, and name it `King_compressed.csv`  
- **Command to execute**:  
  ```bash
  # Create a dataset directory (if it does not exist)
  mkdir -p ./Real_Datasets
  # Download and unzip the dataset
  kaggle datasets download -d harlfoxem/housesalesprediction --unzip -p ./Real_Datasets
  # Rename the dataset (to simplify subsequent loading paths)
  mv ./Real_Datasets/kc_house_data.csv ./Real_Datasets/King_compressed.csv
  ```  


## III. Model Running Examples

### 3.1 Running the Tabula Model (using the Insurance Dataset as an example)
```bash
python Tabula_on_insurance_dataset.py
```


### 3.2 Running the Tabula_middle_padding Model (using the insurance Dataset as an example)
```bash
python Tabula_middle_padding_on_insurance_dataset.py
```

### 3.3 Run Evaluation with evaluate.py
```bash
python evaluation.py
```

# VI. Tabula Reproduction Experiment Results (Insurance Dataset)

| Dataset   | Task Type | Average MAPE (Tabula Reproduction) | Correlation Distance (Tabula Reproduction) |
|-----------|-----------|------------------------------------|-------------------------------------------|
| Insurance | Regression| 0.5288                             | 0.1933                                    |




## V. References
```
@inproceedings{zhao2025tabula,
  title={Tabula: Harnessing language models for tabular data synthesis},
  author={Zhao, Zilong and Birke, Robert and Chen, Lydia Y},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={247--259},
  year={2025},
  organization={Springer}
}
```
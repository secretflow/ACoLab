import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import pandas as pd
from sdv.metadata import SingleTableMetadata

from privacy_utility_framework.privacy_utility_framework.synthesizers.synthesizers import GaussianMixtureModel, \
    GaussianCopulaModel, CTGANModel, CopulaGANModel, TVAEModel, RandomModel

def syn_generation_example():
    # Define the original dataset name
    orig = "insurance"
    # Specify the folder for saving datasets
    folder = f"{orig}_datasets"
    
    syn_save_dir = f"{folder}/syn_on_train"
    os.makedirs(syn_save_dir, exist_ok=True)

    # Flag indicating whether to use the training dataset
    use_train = True

    # Flag indicating whether to generate new synthetic data
    generate_new_syn = True

    # Load the appropriate dataset based on the use_train flag
    if use_train:
        # Load the training dataset from the specified path
        data = pd.read_csv(
            f'../examples/{orig}_datasets/train/{orig}.csv',
            delimiter=',')
        # Adjust the folder path for synthetic data generation
        folder = f"{folder}/syn_on_train"

    else:
        # Load the original dataset if not using training data
        data = pd.read_csv(f'../datasets/original/{orig}.csv', delimiter=',')

    # Generate new synthetic data if the flag is set
    if generate_new_syn:
        # Create metadata for the dataset
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

        # Initialize and fit Gaussian Mixture Model (GMM)
        gmm_model = GaussianMixtureModel(max_components=10)
        gmm_model.fit(data)
        # Save GMM-generated synthetic samples to CSV
        gmm_model.save_sample(f"{folder}/gmm_sample.csv", len(data))

        # Initialize and fit Gaussian Copula Model
        gaussian_copula_model = GaussianCopulaModel(metadata)
        gaussian_copula_model.fit(data)
        # Save Gaussian Copula-generated synthetic samples and model
        gaussian_copula_model.save_sample(f"{folder}/gaussian_copula_sample.csv", len(data))
        gaussian_copula_model.save_model(f"{folder}/gaussian_copula_model.pkl")

        # Initialize and fit CTGAN Model
        ctgan_model = CTGANModel(metadata)
        ctgan_model.fit(data)
        # Save CTGAN-generated synthetic samples and model
        ctgan_model.save_sample(f"{folder}/ctgan_sample.csv", len(data))
        ctgan_model.save_model(f"{folder}/ctgan_model.pkl")

        # Initialize and fit Copula GAN Model
        copulagan_model = CopulaGANModel(metadata)
        copulagan_model.fit(data)
        # Save Copula GAN-generated synthetic samples and model
        copulagan_model.save_sample(f"{folder}/copulagan_sample.csv", len(data))
        copulagan_model.save_model(f"{folder}/copulagan_model.pkl")

        # Initialize and fit TVAE Model
        tvae_model = TVAEModel(metadata)
        tvae_model.fit(data)
        # Save TVAE-generated synthetic samples and model
        tvae_model.save_sample(f"{folder}/tvae_sample.csv", len(data))
        tvae_model.save_model(f"{folder}/tvae_model.pkl")

        # Initialize and fit Random Model
        random_vanilla_model = RandomModel()
        random_vanilla_model.fit(data)
        # Save randomly generated samples to CSV
        random_vanilla_model.save_sample(f"{folder}/random_sample.csv", len(data))

def load_model_example():
    # Load a previously saved CTGAN model from the specified path
    ctgan_model = CTGANModel.load_model(
        "../examples/insurance_datasets/ctgan_model.pkl")
    # Generate samples from the loaded CTGAN model
    samples_from_loaded_model = ctgan_model.sample(10)
    print(f"~~~~~Samples from loaded CTGAN Model~~~~~\n {samples_from_loaded_model}")

# Execute the synthetic data generation example
syn_generation_example()

# Execute the model loading example
load_model_example()

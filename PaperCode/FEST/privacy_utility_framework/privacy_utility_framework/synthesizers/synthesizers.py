from abc import ABC
import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer
from sklearn.mixture import GaussianMixture

from privacy_utility_framework.privacy_utility_framework.dataset.dataset import Dataset


class BaseModel(ABC):
    """
    Abstract base class for models that use a synthesizer to generate synthetic data.

    Attributes:
        synthesizer_class (class): Placeholder for the synthesizer class, to be specified in subclasses.
    """
    synthesizer_class = None

    def __init__(self, synthesizer):
        """
        Initializes the model with a specific synthesizer instance.

        Args:
            synthesizer: An instance of a synthesizer class used for synthetic data generation.
        """
        self.synthesizer = synthesizer

    def fit(self, data: pd.DataFrame) -> None:
        """
        Trains the synthesizer on the provided dataset.

        Args:
            data (pd.DataFrame): The data to be used for fitting the synthesizer.
        """
        self.synthesizer.fit(data)

    def sample(self, num_samples: int = 200) -> pd.DataFrame:
        """
        Generates synthetic samples using the trained synthesizer.

        Args:
            num_samples (int): The number of synthetic samples to generate (default is 200).

        Returns:
            pd.DataFrame: The generated synthetic samples.
        """
        return self.synthesizer.sample(num_samples)

    def save_sample(self, filename: str, num_samples: int = 200) -> None:
        """
        Generates and saves synthetic samples to a CSV file.

        Args:
            filename (str): The name of the file to save the synthetic data.
            num_samples (int): The number of synthetic samples to generate (default is 200).
        """
        synthetic_data = self.sample(num_samples)
        synthetic_data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def save_model(self, filename: str) -> None:
        """
        Saves the current synthesizer model to a file, if supported by the synthesizer.

        Args:
            filename (str): The name of the file to save the model.

        Raises:
            AttributeError: If the synthesizer does not support the .save method.
        """
        if hasattr(self.synthesizer, 'save'):
            self.synthesizer.save(filename)
            print(f"Model saved to {filename}")
        else:
            raise AttributeError(f"The current synthesizer does not support the 'save' method.")

    @classmethod
    def load_model(cls, filepath: str):
        """
        Loads a saved synthesizer model from a specified file and returns a model instance,
        if supported by the synthesizer.

        Args:
            filepath (str): The path to the file containing the saved synthesizer model.

        Returns:
            An instance of the model with the synthesizer loaded from the specified file.

        Raises:
            AttributeError: If the synthesizer class does not support the .load method.
        """
        if hasattr(cls.synthesizer_class, 'load'):
            synthesizer = cls.synthesizer_class.load(filepath)
            instance = cls.__new__(cls)
            instance.synthesizer = synthesizer
            return instance
        else:
            raise AttributeError(
                f"The synthesizer class '{cls.synthesizer_class.__name__}' does not support the 'load' method.")


class GaussianCopulaModel(BaseModel):
    """
    A model for synthetic data generation using the Gaussian Copula method.
    Specifies GaussianCopulaSynthesizer as the synthesizer.
    """
    synthesizer_class = GaussianCopulaSynthesizer

    def __init__(self, metadata: SingleTableMetadata):
        """
        Initializes GaussianCopulaModel with a GaussianCopulaSynthesizer instance.

        Args:
            metadata (SingleTableMetadata): Metadata for the dataset structure for the synthesizer.
        """
        super().__init__(GaussianCopulaSynthesizer(metadata))


class CTGANModel(BaseModel):
    """
    A model for synthetic data generation using the CTGAN approach.
    Specifies CTGANSynthesizer as the synthesizer.
    """
    synthesizer_class = CTGANSynthesizer

    def __init__(self, metadata: SingleTableMetadata):
        """
        Initializes CTGANModel with a CTGANSynthesizer instance.

        Args:
            metadata (SingleTableMetadata): Metadata for the dataset structure for the synthesizer.
        """
        super().__init__(CTGANSynthesizer(metadata))


class CopulaGANModel(BaseModel):
    """
    A model for synthetic data generation using the Copula GAN approach.
    Specifies CopulaGANSynthesizer as the synthesizer.
    """
    synthesizer_class = CopulaGANSynthesizer

    def __init__(self, metadata: SingleTableMetadata):
        """
        Initializes CopulaGANModel with a CopulaGANSynthesizer instance.

        Args:
            metadata (SingleTableMetadata): Metadata for the dataset structure for the synthesizer.
        """
        super().__init__(CopulaGANSynthesizer(metadata))


class TVAEModel(BaseModel):
    """
    A model for synthetic data generation using the TVAE approach.
    Specifies TVAESynthesizer as the synthesizer.
    """
    synthesizer_class = TVAESynthesizer

    def __init__(self, metadata: SingleTableMetadata):
        """
        Initializes TVAEModel with a TVAESynthesizer instance.

        Args:
            metadata (SingleTableMetadata): Metadata for the dataset structure for the synthesizer.
        """
        super().__init__(TVAESynthesizer(metadata))


class GaussianMixtureModel(BaseModel):
    """
    Model for generating synthetic data using a Gaussian Mixture Model (GMM).
    Performs data transformation and fitting with optional selection of optimal components.
    """

    def __init__(self, max_components: int = 10):
        """
        Initializes GaussianMixtureModel with a maximum number of components to test for GMM.

        Args:
            max_components (int): The maximum number of components to consider.
        """
        super().__init__(None)
        self.transformed_data = None
        self.transformer = None
        self.max_components = max_components
        self.model = None

    def fit(self, data: pd.DataFrame, random_state: int = 42) -> None:
        """
        Transforms data, selects optimal components, and fits the GMM.

        Args:
            data (pd.DataFrame): Input data for model fitting.
            random_state (int): Seed for reproducibility.
        """
        dataset = Dataset(data)
        dataset.set_transformer_and_scaler()
        dataset.transform()
        self.transformed_data = dataset.transformed_data
        self.transformer = dataset.transformer
        optimal_n_components = self._select_n_components(self.transformed_data, random_state)
        self.model = GaussianMixture(n_components=optimal_n_components, random_state=random_state)
        self.model.fit(self.transformed_data)

    def sample(self, num_samples: int = 200) -> pd.DataFrame:
        """
        Generates synthetic samples by sampling from the fitted GMM.

        Args:
            num_samples (int): Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: Synthetic samples in the original data format.
        """
        if self.model is not None:
            samples, _ = self.model.sample(num_samples)
            samples_pd = pd.DataFrame(samples, columns=self.transformed_data.columns)
            inverse_samples = self.transformer.reverse_transform(samples_pd)
            return inverse_samples
        else:
            raise RuntimeError("Data has not been fitted yet.")

    # Not defined for the gmm model
    def save_model(self, filename: str) -> None:
        pass
        # NOTE: this was a test of saving the gmm model
        # np.save(filename + '_weights', self.model.weights_, allow_pickle=False)
        # np.save(filename + '_means', self.model.means_, allow_pickle=False)
        # np.save(filename + '_covariances', self.model.covariances_, allow_pickle=False)
        # np.save(filename + '_precisions_cholesky', self.model.precisions_cholesky_, allow_pickle=False)
        # print(f"Model saved to {filename}")

    # Not defined for the gmm model
    @classmethod
    def load_model(cls, filepath: str):
        pass
        # NOTE: this was a test of loading the saved gmm model
        # means = np.load(filepath + '_means.npy')
        # covar = np.load(filepath + '_covariances.npy')
        # loaded_gmm = GaussianMixture(n_components=len(means), covariance_type='full')
        # loaded_gmm.precisions_cholesky_ = np.load(filepath + '_precisions_cholesky.npy')
        # loaded_gmm.weights_ = np.load(filepath + '_weights.npy')
        # loaded_gmm.means_ = means
        # loaded_gmm.covariances_ = covar
        # instance = cls.__new__(cls)  # Create an instance without calling __init__
        # instance.model = loaded_gmm
        # print("Gaussian Mixture Model was loaded.")
        # return instance

    def _select_n_components(self, data: pd.DataFrame, random_state: int) -> int:
        """
        Selects the optimal number of GMM components using the Bayesian Information Criterion (BIC).

        Args:
            data (pd.DataFrame): Dataset for fitting GMM models.
            random_state (int): Seed for reproducibility.

        Returns:
            int: Optimal number of components based on BIC.
        """
        bics = []
        n_components_range = range(1, self.max_components + 1)

        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=random_state)
            gmm.fit(data)
            bics.append(gmm.bic(data))

        optimal_n_components = n_components_range[np.argmin(bics)]
        return optimal_n_components


class RandomModel(BaseModel):
    """
    Model that generates synthetic data by randomly sampling from a given dataset.
    This model does not require complex fitting.
    """

    def __init__(self):
        """
         Initializes RandomModel with no synthesizer and sets default attributes.
         """
        super().__init__(None)
        self.data = None
        self.trained = False

    def fit(self, data: pd.DataFrame) -> None:
        """
        Sets the provided dataset as the source for random sampling.

        Args:
            data (pd.DataFrame): The dataset to be used for random sampling.
        """
        self.trained = True
        self.data = data

    def sample(self, num_samples: int = None, random_state: int = None) -> pd.DataFrame:
        """
        Randomly samples data points from the dataset.

        Args:
            num_samples (int): Number of samples to generate.
            random_state (int): Seed for reproducibility.

        Returns:
            pd.DataFrame: Containing randomly sampled data.

        Raises:
            RuntimeError if the model has not been fitted with a dataset.
        """
        if self.trained:
            if num_samples is None:
                return self.data
            return pd.DataFrame(self.data.sample(num_samples, random_state=random_state, replace=False))
        else:
            raise RuntimeError("No dataset provided to generator")

    # Not defined for the random model
    def save_model(self, filename: str) -> None:
        pass

    # Not defined for the random model
    @classmethod
    def load_model(cls, filepath: str):
        pass

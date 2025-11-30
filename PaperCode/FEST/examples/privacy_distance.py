import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import pandas as pd

from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.disco import \
    DisclosureCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.adversarial_accuracy_class import \
    AdversarialAccuracyCalculator, AdversarialAccuracyCalculator_NN
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.dcr_class import DCRCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.nndr_class import \
    NNDRCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.privacy_metric_manager import \
    PrivacyMetricManager


def dcr_example():
    print("~~~~~~~~~DCR EXAMPLE~~~~~~~~~~")
    original_data = pd.read_csv("../datasets/original/diabetes.csv")
    synthetic_data = pd.read_csv(
        "../datasets/synthetic/diabetes_datasets/ctgan_sample.csv")

    test_dcr_calculator = DCRCalculator(original_data,
                                        synthetic_data, weights=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    test_dcr = test_dcr_calculator.evaluate()
    print(f'DCR (diabetes, ctgan): {test_dcr}')


def nndr_example():
    print("~~~~~~~~~NNDR EXAMPLE~~~~~~~~~~")
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets = ["diabetes", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            test_nndr_calculator = NNDRCalculator(original_data, synthetic_data)
            test_nndr = test_nndr_calculator.evaluate()
            print(f'NNDR {orig, syn}: {test_nndr}')


def nnaa_example():
    print("~~~~~~~~~NNAA EXAMPLE~~~~~~~~~~")
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets = ["diabetes", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            print(f'~~~~~~Adversarial Accuracy CDIST~~~~~~ {orig, syn}')

            calculator_cdist = AdversarialAccuracyCalculator(original_data, synthetic_data)
            nnaa1 = calculator_cdist.evaluate()
            print(nnaa1)

            print(f'~~~~~~Adversarial Accuracy NN~~~~~~ {orig, syn}')
            calculator_nn = AdversarialAccuracyCalculator_NN(original_data, synthetic_data)
            nnaa2 = calculator_nn.evaluate()
            print(nnaa2)


def disco_example():
    print("~~~~~~~~~DiSCO/REPU EXAMPLE~~~~~~~~~~")
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes"]
    diabetes_keys = ['Age', 'BMI', 'DiabetesPedigreeFunction', 'Glucose', 'BloodPressure']
    diabetes_target = 'Outcome'

    # Other examples, adjust parameters below

    # original_datasets =["cardio"]
    # cardio_keys = ['age', 'gender', 'height', 'weight', 'cholesterol', 'gluc']
    # cardio_target = 'cardio'
    #
    #
    original_datasets =["insurance"]
    insurance_keys = ['age', 'bmi', 'children']
    insurance_target = 'charges'

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            print(f"~~~Pair: {orig, syn}~~~\n")
            calc = DisclosureCalculator(original_data, synthetic_data, keys=insurance_keys, target=insurance_target)
            repU, DiSCO = calc.evaluate()
            print(f"repU: {repU}, DiSCO: {DiSCO}")


def privacy_metric_manager_example():
    original_data = pd.read_csv(f"../datasets/original/diabetes.csv")
    synthetic_data = pd.read_csv(
        f"../datasets/synthetic/diabetes_datasets/ctgan_sample.csv")
    original_name = "Diabetes"
    synthetic_name = "CTGAN"
    p = PrivacyMetricManager()
    metric_list = \
        [
            DCRCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
            NNDRCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
            AdversarialAccuracyCalculator(original_data, synthetic_data, original_name=original_name,
                                          synthetic_name=synthetic_name)
        ]
    p.add_metric(metric_list)
    results = p.evaluate_all()
    for key, value in results.items():
        print(f"{key}: {value}")

dcr_example()
nndr_example()
nnaa_example()
privacy_metric_manager_example()
disco_example()

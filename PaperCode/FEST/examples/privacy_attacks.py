import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.attacks.inference_class import \
    InferenceCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.attacks.linkability_class import \
    LinkabilityCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.attacks.singlingout_class import \
    SinglingOutCalculator


def inference_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"./{orig}_datasets/train/{orig}.csv")
            synthetic_data = pd.read_csv(f"./{orig}_datasets/syn_on_train/{syn}_sample.csv")
            control = pd.read_csv(f"./{orig}_datasets/test/{orig}.csv")
            columns = original_data.columns
            results = []

            # Iterate over all columns as secret, the rest as aux_cols and compute the individual risks
            # In the thesis report, the risk with the highest value was chosen and added to the results table
            for secret in columns:
                aux_cols = [col for col in columns if col != secret]
                test_sing = InferenceCalculator(original_data, synthetic_data, aux_cols=aux_cols, secret=secret,
                                                control=control, n_attacks=200)
                t = test_sing.evaluate()
                results.append((secret, t))
            print(f"~~~Synthetic Dataset generated with: {syn}~~~")
            print(results)


def linkability_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance"]
    aux_cols_i = (["age", "sex", "bmi"], ["children", "smoker", "region", "charges"])

    print("STARTED")

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../examples/{orig}_datasets/train/{orig}.csv")
            synthetic_data = pd.read_csv(f"../examples/{orig}_datasets/syn_on_train/{syn}_sample.csv")
            control_orig = pd.read_csv(f"../examples/{orig}_datasets/test/{orig}.csv")
            link = LinkabilityCalculator(original_data, synthetic_data, aux_cols=aux_cols_i, control=control_orig, n_attacks=260)

            result = link.evaluate()
            print(f"~~~Linkability Risk for {orig, syn}~~~")
            print(result)

def singling_out_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../examples/{orig}_datasets/train/{orig}.csv")
            synthetic_data = pd.read_csv(f"../examples/{orig}_datasets/syn_on_train/{syn}_sample.csv")

            test_sing = SinglingOutCalculator(original_data, synthetic_data)

            # NOTE: This may take a little longer
            results = test_sing.evaluate()
            print(f"~~~Singling Out Risk for {orig, syn}~~~")
            print(results)


if __name__ == "__main__":
    inference_example()
    linkability_example()
    singling_out_example()
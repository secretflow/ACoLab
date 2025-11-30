import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.basic_stats import \
    BasicStatsCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.correlation import \
    CorrelationCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.js_similarity import \
    JSCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.ks_test import KSCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.mutual_information import \
    MICalculator
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.wasserstein import \
    WassersteinMethod, WassersteinCalculator
from privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.utility_metric_manager import \
    UtilityMetricManager


def wasserstein_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes"]
    all_wasserstein_distances = {method: [] for method in WassersteinMethod}
    print(all_wasserstein_distances)
    for orig in original_datasets:
        for syn in synthetic_datasets:
            print(f"~~~PAIR: {orig, syn}~~~")
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            for method in all_wasserstein_distances:
                calc = WassersteinCalculator(original_data, synthetic_data)
                res = calc.evaluate(metric=method)
                all_wasserstein_distances[method].append(res)

def mutual_information_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "cardio", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")

            calc = MICalculator(original_data, synthetic_data)
            print(f"~~~Pair: {orig, syn}~~~\n")
            print(calc.evaluate())

def ks_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance", "diabetes"]
    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            calc = KSCalculator(original_data, synthetic_data)
            print(f"~~~Pair: {orig, syn}~~~\n")
            print(calc.evaluate())


def js_similarity_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets = ["cardio", "insurance", "diabetes"]

    for orig in original_datasets:
        original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
        for syn in synthetic_datasets:

            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            calc = JSCalculator(original_data, synthetic_data)
            score = calc.evaluate()
            print(f"~~~Pair: {orig, syn}~~~")
            print(f"JS-Similarity Score: {score}\n")


def correlation_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance", "diabetes"]
    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            calc = CorrelationCalculator(original_data, synthetic_data)
            print(f"~~~Pair: {orig, syn}~~~")
            print(f"{calc.evaluate()}\n")


def basic_stats_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "cardio", "insurance"]
    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            calc = BasicStatsCalculator(original_data, synthetic_data)
            res = calc.evaluate()
            print(f"PAIR {orig, syn}")
            print(res)

def utility_metric_manager_example():
    original_data = pd.read_csv(f"../datasets/original/insurance.csv")
    synthetic_data = pd.read_csv(
        f"../datasets/synthetic/insurance_datasets/ctgan_sample.csv")
    original_name = "Insurance"
    synthetic_name = "CTGAN"
    p = UtilityMetricManager()
    metric_list = \
        [
            BasicStatsCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
            MICalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name),
        ]
    p.add_metric(metric_list)
    results = p.evaluate_all()
    for key, value in results.items():
        print(f"{key}: {value}")

wasserstein_example()
mutual_information_example()
ks_example()
js_similarity_example()
correlation_example()
basic_stats_example()
utility_metric_manager_example()
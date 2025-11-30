# FEST: A Unified Framework for Evaluating Synthetic Tabular Data

This repo is a reproduction of the paper: [FEST: A Unified Framework for Evaluating Synthetic Tabular Data](https://arxiv.org/abs/2508.16254).

This repository is developed based on an existing open-source repository: [synprivutil](https://github.com/Karo2222/synprivutil).


## 1. Project Background

**Core Objective**: Reproduce the FEST framework's "privacy assessment" and "utility assessment" of synthetic data, and verify the "privacy-utility trade-off" relationship.

**Key Outputs**:
- Privacy/utility metrics for 6 synthetic models (CTGAN, GMM, CopulaGAN, etc.)
- 5 types of core visualization plots from the paper 

## 2. Prerequisites 

## Create Virtual Environment

```bash
# Create conda environment
conda create -n fest python=3.10.19
conda activate fest
```

## Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt
```


**Data Foundation**: The 3 core datasets (diabetes.csv, cardio.csv, insurance.csv) exist in the `datasets/original/` directory

## 3. Core Reproduction Workflow 

### Phase 1: Data Preparation (Generate Training/Test Sets)

**Run Script**: `examples/train_test.py`

**Purpose**: Split raw data into a training set (for synthetic data generation) and a test set (for privacy attack control groups)

**Dependencies**: `datasets/original/insurance.csv` (use the insurance dataset as an example; extendable to others)

**Operation**:
```bash
# Navigate to the examples directory
cd synprivutil/examples  
# Run the script
python train_test.py
```

**Expected Output**:
- `insurance_datasets/train/insurance.csv` 
- `insurance_datasets/test/insurance.csv` 



### Phase 2: Generate Synthetic Data

**Run Script**: `examples/synthetic_data_generation.py`

**Purpose**: Use 6 models to generate synthetic data matching the training set size:
- Statistical models: Gaussian Mixture (GMM), Gaussian Copula (GC)
- Deep learning models: CTGAN, CopulaGAN, TVAE
- Baseline: Random Model

**Dependencies**: Training set from Phase 1 (`insurance_datasets/train/insurance.csv`)

**Operation**:
```bash
# Ensure you are in the examples directory
python synthetic_data_generation.py
```


**Expected Output**:
- 6 synthetic data files (e.g., `ctgan_sample.csv`, `gmm_sample.csv`) in `insurance_datasets/syn_on_train/`
- Corresponding model files (e.g., `ctgan_model.pkl`) for re-use



### Phase 3: Data Preprocessing (Run On-Demand)

**Run Script**: `examples/dataset_transform_normalization.py`

**Purpose**: Standardize raw/synthetic data to resolve format issues:
- Encode categorical variables (e.g., sex, smoker)
- Normalize numerical variables (e.g., scale age [18–64] and charges [1k–60k] to the same range)

**Dependencies**:
- Raw data: `datasets/original/insurance.csv`
- Synthetic data: `insurance_datasets/syn_on_train/xxx_sample.csv`

**Operation**:
```bash
# Ensure you are in the examples directory
python dataset_transform_normalization.py
```



### Phase 4: Privacy Assessment (Core Metrics)

Privacy assessment has two sub-types: attack-based (simulate real-world attacks) and distance-based (measure data similarity).

#### 4.1 Attack-Based Privacy Metrics

**Script**: `examples/privacy_attacks.py`

**Metrics Evaluated**:
- **Singling Out Risk**: Risk of identifying a unique individual via attribute combinations
- **Linkability Risk**: Risk of linking data from multiple sources to identify an individual
- **Inference Risk**: Risk of deducing sensitive attributes (e.g., charges) from auxiliary data (e.g., age, bmi)

**Dependencies**: Training/test sets (Phase 1), synthetic data (Phase 2)

**Operation**:
```bash
python privacy_attacks.py
```



**Expected Results (Consistent with Paper Table 8)**:

| Model | Singling Out Risk | Linkability Risk | Inference Risk |
|-------|-------------------|------------------|----------------|
| CTGAN | ~0.11–0.13 | ~0.0–0.03 | ~0.04 |
| GMM | ~0.06–0.12 | ~0.002 | ~0.01 |
| Random | ~0.996 | ~0.989 | ~0.99 |

#### 4.2 Distance-Based Privacy Metrics

**Script**: `examples/privacy_distance.py`

**Metrics Evaluated**:
- **DCR (Distance of Closest Record)**: Average distance between synthetic and raw data points (higher = better privacy)
- **NNDR (Nearest Neighbor Distance Ratio)**: Measure of data point isolation (higher = lower re-identification risk)
- **repU (Replicated Uniques)**: Risk of replicating unique raw data records (lower = better privacy)
- **DiSCO (Disclosive in Synthetic Correct Original)**: Risk of attribute disclosure (lower = better privacy)
- **NNAA (Nearest-Neighbor Adversarial Accuracy)**: Privacy-utility balance (closer to 0.5 = better balance)

**Dependencies**: Raw data (Phase 1), synthetic data (Phase 2); preprocessed data (Phase 3) for accuracy

**Operation**:
```bash
python privacy_distance.py
```


**Expected Results (Consistent with Paper Table 7)**:

| Model | DCR | NNDR | repU | DiSCO | NNAA |
|-------|-----|------|------|-------|------|
| CTGAN | ~0.27 | ~0.83 | 0.0 | 0.0 | ~0.73 |
| GMM | ~0.14 | ~0.77 | 0.0 | 0.0 | ~0.58 |
| Random | 0.0 | 0.0 | ~98.5 | ~98.7 | 0.0 |

### Phase 5: Utility Assessment (Core Metrics)

**Run Script**: `examples/utility.py`

**Purpose**: Verify if synthetic data retains the statistical properties of raw data (critical for downstream ML tasks)

**Metrics Evaluated**:
- **KS Similarity**: Compares cumulative distribution functions (CDFs) of raw/synthetic data (closer to 1 = better)
- **Pearson Correlation**: Measures preservation of variable relationships (closer to 1 = better)
- **Mean/Median/Variance Difference**: Differences in basic statistics (lower = better)
- **Normalized Mutual Information (NMI)**: Measures variable dependence (closer to 1 = better)

**Dependencies**: Raw data (Phase 1), synthetic data (Phase 2)

**Operation**:
```bash
python utility.py
```



**Expected Results (Consistent with Paper Table 9)**:

| Model | KS Similarity | Pearson Correlation | Mean Difference | NMI |
|-------|---------------|---------------------|-----------------|-----|
| GMM | ~0.97 | ~0.99 | ~0.01 | ~0.99 |
| Gaussian Copula | ~0.97 | ~0.97 | ~0.016 | ~0.99 |
| Random | 1.0 | 1.0 | 0.0 | 1.0 |

### Phase 6: Generate Visualization Plots

**Run Script**: `examples/plots.py`

**Purpose**: Generate 5 key plots from the paper to visualize results

**Dependencies**:
- Raw/synthetic data (Phases 1–2)
- Preprocessed data (Phase 3) for pairwise plots

**Operation**:
```bash
# Create a directory to save plots (avoids "no such file" errors)
mkdir -p ../plots  
# Run the plotting script
python plots.py
```



**Expected Outputs (Saved in `../plots/`)**:

| Plot File | Corresponding Paper Figure | Core Content |
|-----------|---------------------------|--------------|
| `basic_stats_mean_diabetes_plot.png` | Figure 3 | Mean comparison (raw vs. synthetic data) |
| `wasserstein_diabetes_plot.png` | Figure 4 | Wasserstein distance (distribution similarity) |
| `correlation_heatmap_cardio_tvae.png` | Figure 5 | Correlation heatmap (variable relationships) |
| `ks_similarity_insurance_plot.png` | Figure 6 | KS similarity bar chart |
| `mi_heatmap_insurance_copulagan.png` | Figure 7 | Mutual information heatmap (variable dependence) |



## 4. Result Verification 

Reproduction is successful if all of the following are true:

- **Metrics Match the Paper**: Privacy/utility metrics align with Tables 3, 6, 7, 8, and 9
- **Plots Are Valid**: Visualizations in `../plots/` match the trends of Figures 3–7 (e.g., synthetic data means overlap with raw data)
- **Key Conclusion Holds**:
  - Generative models (CTGAN, GMM) balance privacy and utility
  - The Random model has perfect utility but no privacy protection
  - GMM achieves the best overall trade-off (NNAA ≈ 0.58, KS ≈ 0.97)

## Citing FEST

```
@misc{niu2025festunifiedframeworkevaluating,
      title={FEST: A Unified Framework for Evaluating Synthetic Tabular Data}, 
      author={Weijie Niu and Alberto Huertas Celdran and Karoline Siarsky and Burkhard Stiller},
      year={2025},
      eprint={2508.16254},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.16254}, 
}
```
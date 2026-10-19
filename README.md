# DHGCN: A Topology‑Driven Multivariate Entity Association Identification Model for Drug Repositioning via Decoupled Hypergraph Convolution

## Requirements

```
python>=3.9
torch>=2.0
torch_geometric
numpy
pandas
scipy
matplotlib
```

## Data Description

- `Data/`: Standard original dataset, contains train_14400.csv, val_14400.csv, test_14400.csv.
- `cold_start_Data/`: Datasets for cold‑start settings, includes drug_cold, disease_cold, double_cold splits.
- `imbalanced_test_Data/`: Pre‑processed test subsets under different positive‑negative ratios for class imbalance analysis.

## Usage

### Main Model

```
python main.py
```

### Validation with a Parameter‑free Rule Baseline

```
python rule_baseline.py
```

### Cold‑start Experiments

Modify dataset file paths inside main.py to point to corresponding folders under cold_start_Data before running.

```
python main.py
```

### Class Imbalance Sensitivity Analysis

```
python run_sensitivity.py
```

### Mode‑level Attribution

```
python run_attribution.py
```

### Perturbation Analysis

```
python run_perturbation.py
```

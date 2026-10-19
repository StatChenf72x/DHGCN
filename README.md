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

- `Data/`: Standard original dataset, contains train_14400.csv, val_14400.csv, test_14400.csv. Used for main model, rule baseline, attribution and perturbation analysis.
- `cold_start_Data/`: Datasets for cold‑start settings, includes drug_cold, disease_cold, double_cold splits. Only used for cold‑start experiments.
- `imbalanced_test_Data/`: Pre‑processed test subsets under different positive‑negative ratios. Specially used for class imbalance sensitivity analysis.

## Usage

### Main Model

This is the training and evaluation script for the full DHGCN model. It uses the standard dataset from `Data/`.

```
python main.py
```

### Validation with a Parameter‑free Rule Baseline

Parameter‑free heuristic baseline without model training. It uses standard dataset from `Data/`.

```
python rule_baseline.py
```

### Cold‑start Experiments

Evaluate model generalization performance on unseen entities. You need to modify the dataset file paths inside main.py to point to corresponding sub‑folders under `cold_start_Data/` before execution. Reuse main.py for training and evaluation.

```
python main.py
```

### Class Imbalance Sensitivity Analysis

Evaluate model robustness under varying positive‑negative label ratios. Uses pre‑generated test subsets from `imbalanced_test_Data/`.

```
python run_sensitivity.py
```

### Mode‑level Attribution

Interpret model prediction by analyzing semantic mode‑level attention weights. Uses standard dataset from `Data/`, requires pre‑trained DHGCN model weights.

```
python run_attribution.py
```

### Perturbation Analysis

Quantify the contribution of drug, gene and disease components to model prediction. Uses standard dataset from `Data/`, requires pre‑trained DHGCN model weights.

```
python run_perturbation.py
```

## Childcare Institution Layout Forecasting in Shanghai (LightGBM + Two-Layer Potential Demand Model)
(Project Done in 2023. Migrated from previous account)

This repository contains code and assets for forecasting suitable locations of childcare institutions in Shanghai using a machine-learning outer model (LightGBM) and an inner potential-demand model. The study divides the city into 127,568 grid cells at a 250 m × 250 m resolution and constructs features from five perspectives: urban functional structure, natural environment, transportation accessibility, socioeconomic context, and competitive environment.

Key findings from the accompanying technical report:

- **Outer model performance**: Accuracy ≈ 92.9%, **AUC ≈ 0.95** (5-fold CV with LightGBM).
- **Priority sites**: The inner potential-demand model selects **150 priority grid cells**, mostly in areas with high 0–1 age population density and currently no childcare institutions.
- **Top contributors (SHAP/feature importance)**: Kindergarten count, science & education POIs, and residential kernel density.

### Repository Structure (selected)

- `dataset.xlsx`: Raw/working dataset used by preprocessing scripts.
- `dataset_after_missing_values.csv`: Output after missing-value handling.
- `final_dataset.csv`: One-hot encoded and min–max normalized dataset.
- `dataset/` and `dataset1/`: Model prediction outputs across multiple runs (`*.csv`), plus `sum.csv`, `auc.csv`.
- `figure/`: Figures used in the report (AUC/ROC curves, POI kernel density mosaics, etc.).

Core scripts:

- `missing_values_processing.py`: Fills missing values for numeric POI features using mean; fills `tudi` using mode. Outputs `dataset_after_missing_values.csv`.
- `onehot_and_normalization.py`: One-hot encodes `tudi` and applies min–max normalization. Outputs `final_dataset.csv`.
- `gb.py` / `gb_pre_filter.py`: Trains LightGBM classifiers with class rebalancing and 5-fold CV; writes per-run predictions to `dataset1/` or `dataset/`, aggregates AUCs to `auc.csv` and ensemble sums to `sum.csv`.
- `model_reliability.py`: Computes ROC/AUC, plots ROC, and draws a confusion matrix for a held-out test split.
- `shap_analysis.py`: Trains LightGBM and prints feature importances (SHAP can be extended as needed).
- `poi_kernel_density_merge.py`: Merges category POI kernel density images into a 3×5 mosaic figure.
- `outlier_processing.py`: Produces box plots for selected variables to visualize outliers.
- `childcare_kernel_density_processing.py`: Normalizes childcare kernel density CSV (min–max scaling).
- `main.py`: Auxiliary data reshaping by geographic grid (`jingdu`/`weidu`).

### Requirements

- Python 3.8+
- Python packages:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `lightgbm`
  - `matplotlib`
  - `shap` (for interpretation; optional)

Optional external software:

- ArcGIS (used in the report to generate kernel density maps; not required to train/evaluate the ML model).

### Quick Start

1) Prepare data

- Place `dataset.xlsx` at the repository root. Expected fields include basic geography (`jingdu`, `weidu`), binary target `feature`, POI-derived variables (e.g., `gongsi_poi`, `gouwu_poi`, `gonggong_poi`, `fengjing_poi`, `canyin_poi`, `tiyu_poi`, `shenghuo_poi`, `shangwu_poi`, `kejiao_poi`, `jinrong_poi`, `jiaotong_poi`, `zhusu_poi`, `zhengfu_poi`, `yiliao_poi`), environmental/socioeconomic variables (e.g., `gaocheng_mean`, `podu_mean`, `gdp_yuan`, `xiaoqu_dict`, `gongsi_dist`), and categorical land-use `tudi`.

2) Handle missing values

```bash
python missing_values_processing.py
```

This produces `dataset_after_missing_values.csv`.

3) (Optional) One-hot encode and normalize

```bash
python onehot_and_normalization.py
```

This produces `final_dataset.csv` (min–max normalized with one-hot encoded `tudi`). The LightGBM training scripts operate on `dataset_after_missing_values.csv` directly and perform their own sampling; use `final_dataset.csv` if you need normalized inputs for other analysis.

4) Train LightGBM models and generate predictions

```bash
python gb.py
# or
python gb_pre_filter.py
```

These scripts:

- Balance classes by sub-sampling the negative class.
- Perform 5-fold cross-validation and record **AUC** per run.
- Save per-run predictions under `dataset1/` (or `dataset/`) and aggregate AUCs to `auc.csv`. An ensemble sum is saved as `sum.csv`.

5) Evaluate model reliability

```bash
python model_reliability.py
```

This computes ROC/AUC and produces a confusion matrix (`matrix.jpg`).

6) (Optional) Interpretability and visuals

```bash
python shap_analysis.py
python outlier_processing.py
python poi_kernel_density_merge.py
```

- Inspect printed feature importances (extend to full SHAP plots as needed).
- Generate outlier diagnostics and merge POI kernel density figures located in `figure/shanghai_poi_kernel_density/1/`.

### Results (from the report)

- Grid size: 250 m × 250 m; total grid cells: 127,568.
- LightGBM outer model: ~92.9% accuracy; **AUC ≈ 0.95**.
- Inner potential-demand model: **150** priority grid cells concentrated in areas with high 0–1 population density and currently no childcare institutions.
- Important drivers: kindergarten count, science & education POIs, residential kernel density.

### Data Sources (examples from the report/self-doc)

- Shanghai Child Care Service Information Management Platform (for institution POIs).
- Shanghai Open Data Platform (`https://data.sh.gov.cn/`).
- National Floating Population Monitoring Survey datasets (2017, 2018). 


### Notes

- Some figures are generated with ArcGIS (kernel density maps). The machine learning workflow is fully reproducible with Python scripts provided here.
- Folder names and certain variables are in Chinese; this README maps them to English concepts wherever possible.

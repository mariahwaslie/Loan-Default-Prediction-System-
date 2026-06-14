# Loan Default Risk Modeling

This project builds a machine learning pipeline for predicting LendingClub loan defaults from historical borrower, loan, and credit attributes. The work moves from raw loan data through cleaning, leakage prevention, feature engineering, temporal validation, model training, probability calibration, threshold tuning, and error analysis.

The final deliverable is a calibrated default-risk model saved in `models/loan_default_model.pkl`. That artifact includes the trained model, the selected feature list, the categorical feature list, and the optimized decision threshold needed for inference.

## Project Highlights

- Built on a large LendingClub-style dataset with about 1.27 million modeled loans.
- Converts completed loan outcomes into a binary `default` target: `Charged Off = 1`, `Fully Paid = 0`.
- Uses time-based validation so the model is trained on older loans and tested on newer loans, which better reflects real lending deployment.
- Removes post-origination leakage fields such as payment history, recoveries, future FICO pulls, hardship outcomes, and settlement information.
- Trains gradient-boosted models, with the final packaged model using CatBoost plus isotonic calibration.
- Uses SHAP feature importance to reduce the model from 93 candidate features to 75 stronger predictors.
- Optimizes the classification threshold for F1 score to better handle class imbalance.
- Includes calibration analysis, loan-grade subgroup analysis, failure-mode analysis, and bootstrapped confidence intervals.

## Final Model Performance

The final calibrated model was evaluated on the holdout test period after temporal splitting.

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.7531 |
| PR-AUC | 0.3506 |
| Log Loss | 0.3754 |
| Brier Score | 0.1151 |
| Precision | 0.3095 |
| Recall | 0.5732 |
| F1 Score | 0.4020 |
| Optimized Threshold | 0.2625 |

Bootstrap uncertainty estimates from the final notebook:

| Metric | 95% Confidence Interval |
| --- | --- |
| ROC-AUC | 0.748 to 0.759 |
| PR-AUC | 0.341 to 0.362 |

The model favors recall at the selected threshold, which is appropriate for a credit-risk screen where missed defaults can be more costly than additional review.

## Repository Structure

```text
.
|-- data/
|   |-- archive/
|   |   |-- accepted_2007_to_2018Q4.csv
|   |   `-- rejected_2007_to_2018Q4.csv
|   |-- lendingclub_model.csv
|   |-- lendingclub_model_cleaned.csv
|   |-- dti_lendingclub.csv
|   `-- dti_missing_lendingclub.csv
|-- models/
|   |-- loan_default_model.pkl
|   |-- catboost_model.pkl
|   |-- lightgbm_model.pkl
|   |-- lightgbm_iso_model.pkl
|   |-- lgbm_regression_model.pkl
|   |-- regression_feature_columns.pkl
|   |-- cat_cols_reg.pkl
|   `-- preprocess_config.pkl
|-- notebooks/
|   |-- loan-default-prediction.ipynb
|   |-- dti_prediction.ipynb
|   |-- data_leackage_clean.ipynb
|   |-- investigate.ipynb
|   `-- final_model.ipynb
|-- src/
|   |-- functions.py
|   |-- preprocessing.py
|   |-- inference.py
|   `-- explainability.py
`-- tests/
    `-- test_model.py
```

`src/functions.py` currently contains reusable missing-data utilities. The other `src` files, `api/`, `dashboard/`, and `tests/test_model.py` are scaffolded for future packaging, deployment, and test coverage.

## How The Pipeline Works

### 1. Raw Data Preparation

`notebooks/loan-default-prediction.ipynb` starts from the accepted LendingClub data, performs exploratory cleaning, and creates the modeling table.

Key steps:

- Keeps only completed loans with known outcomes: `Fully Paid` and `Charged Off`.
- Creates the binary target column `default`.
- Converts date-like fields into month and year components.
- Handles categorical fields, missing indicators, and numeric cleanup.
- Writes the first modeling dataset to `data/lendingclub_model.csv`.

### 2. DTI Cleaning And Imputation

`notebooks/dti_prediction.ipynb` focuses on debt-to-income ratio (`dti`), an important borrower-risk feature.

Key steps:

- Flags corrupt DTI values with `dti_is_corrupt`.
- Treats DTI values below `0` or above `100` as invalid.
- Splits rows with known DTI from rows requiring imputation.
- Trains and compares Ridge, LightGBM, and CatBoost regressors.
- Saves LightGBM-based DTI imputation artifacts:
  - `models/lgbm_regression_model.pkl`
  - `models/regression_feature_columns.pkl`
  - `models/cat_cols_reg.pkl`
  - `models/preprocess_config.pkl`

The DTI experiments showed that tree-based models outperformed the linear baseline. Validation results in the notebook include Ridge RMSE around `7.06`, LightGBM RMSE around `6.07`, and CatBoost RMSE around `4.75`.

### 3. Leakage Removal

`notebooks/data_leackage_clean.ipynb` removes variables that would not be available at loan-origination time.

Examples of removed leakage fields:

- `loan_status`
- `out_prncp`, `out_prncp_inv`
- `total_pymnt`, `total_rec_prncp`, `total_rec_int`
- `recoveries`, `collection_recovery_fee`
- `last_pymnt_*`, `next_pymnt_*`
- `last_fico_range_low`, `last_fico_range_high`
- hardship and settlement outcome fields

The cleaned dataset is saved as `data/lendingclub_model_cleaned.csv` with about 1.27 million rows and 147 columns.

### 4. Model Investigation

`notebooks/investigate.ipynb` compares multiple classifiers and feature handling strategies.

Models explored include:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

This notebook establishes the baseline modeling direction and shows why gradient-boosted tree models are a good fit for the mix of numeric, categorical, and missingness-driven credit features.

### 5. Final Model Training

`notebooks/final_model.ipynb` contains the production-style modeling workflow.

Key steps:

- Loads `data/lendingclub_model_cleaned.csv`.
- Builds `issue_date` and sorts loans chronologically.
- Uses a temporal split:
  - Train: June 2012 through December 2016
  - Validation: 2017
  - Test: 2018
- Removes highly correlated redundant features.
- Trains CatBoost with tuned hyperparameters:
  - `iterations=1000`
  - `learning_rate=0.1`
  - `depth=6`
  - `l2_leaf_reg=3`
  - `eval_metric="PRAUC"`
- Uses SHAP to remove the least important features.
- Calibrates probabilities with isotonic calibration.
- Chooses threshold `0.2625` based on F1 score.
- Saves the packaged model artifact to `models/loan_default_model.pkl`.

## Saved Model Artifacts

| File | Purpose |
| --- | --- |
| `models/loan_default_model.pkl` | Main deployable artifact containing calibrated CatBoost model, feature list, categorical columns, and threshold |
| `models/catboost_model.pkl` | CatBoost classifier artifact |
| `models/lightgbm_model.pkl` | LightGBM classifier experiment artifact |
| `models/lightgbm_iso_model.pkl` | Isotonic-calibrated LightGBM classifier artifact |
| `models/lgbm_regression_model.pkl` | DTI imputation model |
| `models/regression_feature_columns.pkl` | Feature order for the DTI model |
| `models/cat_cols_reg.pkl` | Categorical columns for the DTI model |
| `models/preprocess_config.pkl` | DTI preprocessing bounds and dropped columns |

## Quick Inference Example

```python
import joblib

bundle = joblib.load("models/loan_default_model.pkl")

model = bundle["model"]
features = bundle["features"]
cat_features = bundle["cat_features"]
threshold = bundle["threshold"]

X_scoring = new_loans[features].copy()

for col in cat_features:
    X_scoring[col] = X_scoring[col].astype("category")

default_probability = model.predict_proba(X_scoring)[:, 1]
default_prediction = (default_probability >= threshold).astype(int)
```

`default_probability` is the calibrated estimated probability of default. `default_prediction` applies the project threshold selected during validation.

## Running The Project

This repository does not currently include a `requirements.txt`, but the notebooks use the following core libraries:

```bash
pip install pandas numpy scikit-learn lightgbm catboost xgboost shap optuna matplotlib seaborn joblib pytest
```

Recommended notebook order:

1. `notebooks/loan-default-prediction.ipynb`
2. `notebooks/dti_prediction.ipynb`
3. `notebooks/data_leackage_clean.ipynb`
4. `notebooks/investigate.ipynb`
5. `notebooks/final_model.ipynb`

The CSV files are large and are ignored by `.gitignore`, so a fresh clone will need access to the LendingClub data or the prepared local CSVs before the notebooks can be rerun end to end.

The test suite is designed to run from saved model artifacts without loading the large CSV files:

```bash
python -m pytest tests/test_model.py -q
```

## Recruiter Notes

This project demonstrates an end-to-end applied machine learning workflow, not just model fitting. The strongest parts are the temporal validation design, explicit leakage removal, class-imbalance handling, probability calibration, model interpretability with SHAP, and performance analysis across loan grades and error types.

Future improvements would be to move notebook logic into the scaffolded `src/preprocessing.py`, `src/inference.py`, and `src/explainability.py` modules; add automated tests in `tests/test_model.py`; and expose the packaged model through the empty `api/` or `dashboard/` directories.

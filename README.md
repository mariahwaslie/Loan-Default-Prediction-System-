# Loan default prediction system

A full-stack machine learning project for predicting the probability that a LendingClub-style loan will default. The project starts with notebook-based data preparation and model research, then packages the final calibrated model behind a FastAPI service with a React/Vite interface and Docker support.

The strongest part of this project is the modeling process behind the app: the workflow uses time-based validation, removes post-origination leakage, calibrates predicted probabilities, tunes the classification threshold for an imbalanced target, and saves a deployment-ready artifact that the API can load directly.

## What this project demonstrates

- End-to-end applied machine learning workflow, from raw loan records to a served prediction endpoint.
- Credit-risk modeling practices, including leakage prevention, temporal validation, class-imbalance handling, probability calibration, threshold tuning, and SHAP-based feature review.
- Production-minded model packaging with a saved inference bundle containing the trained model, feature list, categorical feature list, and decision threshold.
- FastAPI backend that validates requests, loads the model once per process, prepares one-row scoring frames, and returns default probability plus a binary risk prediction.
- React/Vite frontend that lets a user adjust borrower and loan inputs through sliders and call the live prediction API.
- Docker and Docker Compose setup for running the backend and frontend together.
- Automated tests for model artifacts, inference behavior, feature ordering, leakage exclusions, preprocessing utilities, and API route contracts.

## Tech stack

| Area | Tools |
| --- | --- |
| Modeling | Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, XGBoost, SHAP, Optuna |
| API | FastAPI, Pydantic, Uvicorn, joblib |
| Frontend | React, TypeScript, Vite, Axios |
| Packaging | Docker, Docker Compose |
| Testing and CI | pytest, GitHub Actions |

## Repository layout

```text
.
|-- api/
|   |-- main.py                 # FastAPI application setup and CORS configuration
|   `-- routes/
|       `-- predict.py          # health, root, and prediction routes
|-- frontend/
|   |-- src/
|   |   |-- App.tsx             # app shell
|   |   |-- api.ts              # Axios client configured with VITE_API_URL
|   |   |-- defaults.ts         # default scoring payload
|   |   `-- components/
|   |       `-- LoanForm.tsx    # interactive loan-risk form
|   |-- Dockerfile
|   `-- package.json
|-- models/
|   |-- loan_default_model.pkl  # deployable calibrated model bundle
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
|   |-- default_features.py     # 75-feature default loan payload
|   |-- functions.py            # missing-data helper utilities
|   |-- inference.py            # model loading and prediction logic
|   |-- logging_config.py       # application logging setup
|   |-- preprocessing.py        # feature alignment and categorical casting
|   `-- schemas.py              # Pydantic loan request schema
|-- tests/
|   |-- test_api.py
|   |-- test_inference.py
|   |-- test_model.py
|   |-- test_preprocessing.py
|   `-- docker_test.py
|-- Dockerfile                  # backend image
|-- docker-compose.yml          # backend plus frontend development stack
|-- requirements.txt
`-- .github/
    `-- workflows/
        `-- ci.yml
```

## Modeling workflow

### 1. Raw loan preparation

`notebooks/loan-default-prediction.ipynb` builds the initial modeling table from LendingClub-style accepted loan data.

Key steps:

- Keeps completed loans with known outcomes.
- Converts `Charged Off` loans to the positive class and `Fully Paid` loans to the negative class.
- Builds the binary `default` target.
- Cleans numeric, categorical, and date-like fields.
- Writes the prepared modeling dataset to `data/lendingclub_model.csv`.

### 2. DTI cleaning and imputation

`notebooks/dti_prediction.ipynb` focuses on the debt-to-income ratio, a major borrower-risk feature.

Key steps:

- Flags corrupt DTI values with `dti_is_corrupt`.
- Treats DTI values below `0` or above `100` as invalid.
- Splits known-DTI rows from rows requiring imputation.
- Compares Ridge, LightGBM, and CatBoost regressors.
- Saves the LightGBM DTI imputation artifacts used by the project.

The notebook results show tree models outperforming the linear baseline. Reported validation RMSE values were approximately `7.06` for Ridge, `6.07` for LightGBM, and `4.75` for CatBoost.

### 3. Leakage removal

`notebooks/data_leackage_clean.ipynb` removes fields that would not be available at loan-origination time.

Examples of removed leakage columns:

- `loan_status`
- `out_prncp` and `out_prncp_inv`
- `total_pymnt`, `total_rec_prncp`, and `total_rec_int`
- `recoveries` and `collection_recovery_fee`
- `last_pymnt_*` and `next_pymnt_*`
- `last_fico_range_low` and `last_fico_range_high`
- hardship and settlement outcome fields

The cleaned local dataset is saved as `data/lendingclub_model_cleaned.csv` with roughly 1.27 million modeled loans and 147 columns.

### 4. Model comparison

`notebooks/investigate.ipynb` compares several classifier families before selecting the final approach.

Models explored:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Gradient-boosted tree models were a strong fit because the data mixes numeric fields, categorical fields, missingness patterns, and nonlinear credit-risk relationships.

### 5. Final model training

`notebooks/final_model.ipynb` contains the final training and evaluation workflow.

Key steps:

- Loads `data/lendingclub_model_cleaned.csv`.
- Builds `issue_date` and sorts loans chronologically.
- Uses a temporal split that better mirrors lending deployment:
  - Training: June 2012 through December 2016
  - Validation: 2017
  - Test: 2018
- Removes highly correlated redundant features.
- Tunes CatBoost over depth, learning rate, iteration count, and L2 regularization.
- Trains the selected CatBoost model with:
  - `iterations=1000`
  - `learning_rate=0.1`
  - `depth=6`
  - `l2_leaf_reg=3`
  - `eval_metric="PRAUC"`
- Uses SHAP analysis to reduce the model from 93 candidate predictors to 75 final features.
- Calibrates probabilities with isotonic regression through `CalibratedClassifierCV`.
- Selects the decision threshold `0.2625` based on F1 score.
- Saves the deployable bundle to `models/loan_default_model.pkl`.

## Final model performance

The final calibrated CatBoost model was evaluated on the 2018 holdout test period.

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

| Metric | 95% confidence interval |
| --- | --- |
| ROC-AUC | [0.748, 0.759] |
| PR-AUC | [0.341, 0.362] |

At the selected threshold, the model favors recall over precision. That tradeoff is intentional for a screening workflow where missing likely defaults can be more costly than sending additional loans to manual review.

The final notebook also includes calibration analysis, loan-grade subgroup analysis, failure-mode analysis, and bootstrap checks for metric stability. The test default rate was about 14.93%, so PR-AUC and threshold tuning were more informative than accuracy alone.

## Inference pipeline

The serving code turns an API request into a model-ready row in a few explicit steps:

1. `api/routes/predict.py` receives a `POST /predict` request and validates it with the Pydantic `LoanRequest` schema.
2. `src/inference.py` loads `models/loan_default_model.pkl` once at process startup. The path can be overridden with `MODEL_PATH`.
3. `src/preprocessing.py` starts from `BASE_LOAN`, overrides values supplied by the request, aligns the columns to the saved feature list, and casts categorical columns to pandas `category` dtype.
4. The calibrated model returns a default probability.
5. The saved threshold converts that probability into a binary prediction.

The API response contains:

```text
default_probability: calibrated probability of default, from 0 to 1
prediction: 1 when probability >= 0.2625, otherwise 0
```

## API usage

Start the backend, then call the prediction endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 28000,
    "term": 60,
    "int_rate": 25.0,
    "grade": "F",
    "sub_grade": "F4",
    "annual_inc": 38000,
    "dti": 38.0,
    "fico_range_low": 660,
    "revol_util": 92.0,
    "bc_util": 95.0
  }'
```

You can also send `{}` as the payload. The backend will use the default 75-feature loan profile from `src/default_features.py`.

Useful endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic service status |
| `GET` | `/health` | Health check used by tests and Docker smoke checks |
| `POST` | `/predict` | Returns default probability and binary prediction |
| `GET` | `/docs` | FastAPI-generated API documentation |

## Frontend

The React frontend provides an interactive loan-risk form. It exposes sliders for the fields a demo user is most likely to understand quickly, including loan amount, interest rate, annual income, DTI, FICO score, open accounts, revolving utilization, total accounts, bankcard utilization, and mortgage accounts.

The frontend sends the full default loan payload from `frontend/src/defaults.ts`, applies slider overrides, and posts the payload to the FastAPI `/predict` endpoint through the Axios client in `frontend/src/api.ts`.

## Run with Docker Compose

From the repository root:

```bash
docker compose up --build
```

Then open:

- Frontend: `http://localhost:5173`
- Backend health check: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`

The Compose file runs:

- `stats-ml-backend` on port `8000`
- `stats-ml-frontend` on port `5173`

`VITE_API_URL` is set to `http://localhost:8000` for the frontend container.

## Run locally without Docker

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn api.main:app --reload
```

The backend expects the deployable artifact at `models/loan_default_model.pkl`. To use a different model location:

```bash
MODEL_PATH=/path/to/loan_default_model.pkl uvicorn api.main:app --reload
```

### Frontend

In a second terminal:

```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

## Tests

Run the Python test suite from the repository root:

```bash
python -m pytest -q
```

The tests cover:

- Expected model artifacts exist and are non-empty.
- The final model bundle exposes the deployment contract: `model`, `features`, `cat_features`, and `threshold`.
- Saved model features exclude known post-origination leakage columns.
- The model returns valid probabilities and binary predictions.
- A deliberately riskier borrower profile scores higher than a lower-risk profile.
- Feature validation preserves training-time column order.
- Categorical columns are cast correctly for CatBoost inference.
- FastAPI app discovery, health route behavior, and `/predict` route presence.

There is also a live HTTP smoke test in `tests/docker_test.py`. Run it after the API is already serving on `localhost:8000`:

```bash
python -m pytest tests/docker_test.py -q
```

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`.

The workflow:

- Sets up Python 3.11.
- Installs dependencies from `requirements.txt`.
- Runs `pytest -q`.
- Builds the backend Docker image.
- Starts the API container.
- Calls `/health` and `/predict` with `curl`.

## Saved artifacts

| File | Purpose |
| --- | --- |
| `models/loan_default_model.pkl` | Main deployable artifact containing the calibrated CatBoost model, feature list, categorical columns, and threshold |
| `models/catboost_model.pkl` | Uncalibrated CatBoost classifier artifact |
| `models/lightgbm_model.pkl` | LightGBM classifier experiment artifact |
| `models/lightgbm_iso_model.pkl` | Isotonic-calibrated LightGBM classifier artifact |
| `models/lgbm_regression_model.pkl` | DTI imputation model |
| `models/regression_feature_columns.pkl` | Feature order for the DTI imputation model |
| `models/cat_cols_reg.pkl` | Categorical columns for the DTI imputation model |
| `models/preprocess_config.pkl` | DTI preprocessing configuration, including valid DTI bounds and dropped columns |

## Data notes

The CSV files are intentionally ignored by Git because the raw and prepared LendingClub data files are large. A fresh clone can run the API and frontend from the committed model artifacts, but rerunning the notebooks requires local access to the original or prepared data files under `data/`.

Expected local data files include:

- `data/archive/accepted_2007_to_2018Q4.csv`
- `data/archive/rejected_2007_to_2018Q4.csv`
- `data/lendingclub_model.csv`
- `data/lendingclub_model_cleaned.csv`
- `data/dti_lendingclub.csv`
- `data/dti_missing_lendingclub.csv`

## Recruiter summary

This project is more than a modeling notebook. It shows the full path from messy credit data to a usable prediction system: cleaning, leakage control, temporal validation, model selection, calibration, threshold optimization, model packaging, API serving, frontend integration, Dockerization, and tests.

The project also shows judgment around evaluation. Instead of optimizing for accuracy on an imbalanced dataset, it uses PR-AUC, calibration quality, F1-based thresholding, confusion-matrix analysis, and bootstrap confidence intervals to explain how the model behaves under realistic lending conditions.

## Limitations and next steps

This is a portfolio and educational project, not a production credit-decisioning system. Before real-world use, it would need fairness analysis, compliance review, model-monitoring infrastructure, drift detection, retraining automation, stronger API validation, and clear policies for adverse-action explanations.

Good next engineering steps would be:

- Move more of the notebook training logic into versioned Python modules.
- Add a reproducible training CLI.
- Expand the frontend to expose all 75 model features through grouped input sections.
- Add richer response details, such as top contributing risk factors from SHAP.
- Add request-level validation ranges for fields such as DTI, FICO, utilization, and loan amount.
- Add model cards and dataset documentation for governance.

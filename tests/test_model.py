import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

joblib = pytest.importorskip("joblib")

from src.functions import drop_cols_by_missing, drop_rows_by_missing


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

DEFAULT_MODEL_PATH = MODELS_DIR / "loan_default_model.pkl"
DTI_MODEL_PATH = MODELS_DIR / "lgbm_regression_model.pkl"
DTI_FEATURES_PATH = MODELS_DIR / "regression_feature_columns.pkl"
DTI_CAT_COLS_PATH = MODELS_DIR / "cat_cols_reg.pkl"
DTI_CONFIG_PATH = MODELS_DIR / "preprocess_config.pkl"


LEAKAGE_COLUMNS = {
    "loan_status",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_low",
    "last_fico_range_high",
    "hardship_loan_status",
    "settlement_status",
    "debt_settlement_flag_date",
    "settlement_date",
}


DEFAULT_LOAN_VALUES = {
    "loan_amnt": 12000.0,
    "term": 36,
    "int_rate": 13.5,
    "grade": "C",
    "sub_grade": "C3",
    "emp_length": 5,
    "home_ownership": "RENT",
    "annual_inc": 65000.0,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "zip_code": 60600,
    "dti": 18.4,
    "delinq_2yrs": 0,
    "fico_range_low": 690,
    "inq_last_6mths": 1,
    "mths_since_last_delinq": 0,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": 8000,
    "revol_util": 55.0,
    "total_acc": 24,
    "initial_list_status": 1,
    "collections_12_mths_ex_med": 0,
    "verification_status_joint": "NONE",
    "tot_coll_amt": 0,
    "open_acc_6m": 1,
    "open_act_il": 2,
    "open_il_12m": 1,
    "open_il_24m": 2,
    "total_bal_il": 16000,
    "il_util": 70.0,
    "open_rv_12m": 2,
    "open_rv_24m": 4,
    "max_bal_bc": 3000,
    "all_util": 62.0,
    "total_rev_hi_lim": 15000,
    "inq_fi": 1,
    "total_cu_tl": 0,
    "inq_last_12m": 2,
    "acc_open_past_24mths": 4,
    "avg_cur_bal": 9000,
    "bc_open_to_buy": 5000,
    "bc_util": 58.0,
    "mo_sin_old_il_acct": 120,
    "mo_sin_old_rev_tl_op": 160,
    "mo_sin_rcnt_rev_tl_op": 8,
    "mort_acc": 1,
    "mths_since_recent_bc": 8,
    "mths_since_recent_inq": 3,
    "mths_since_recent_revol_delinq": 0,
    "num_accts_ever_120_pd": 0,
    "num_actv_bc_tl": 3,
    "num_actv_rev_tl": 5,
    "num_bc_sats": 4,
    "num_bc_tl": 8,
    "num_il_tl": 6,
    "num_op_rev_tl": 7,
    "num_rev_accts": 14,
    "num_tl_90g_dpd_24m": 0,
    "num_tl_op_past_12m": 2,
    "pct_tl_nvr_dlq": 95.0,
    "percent_bc_gt_75": 25.0,
    "pub_rec_bankruptcies": 0,
    "tot_hi_cred_lim": 180000,
    "total_bal_ex_mort": 26000,
    "total_bc_limit": 11000,
    "total_il_high_credit_limit": 23000,
    "issue_d_month": 6,
    "latitude": 41.88,
    "longitude": -87.63,
    "earliest_cr_month": 8,
    "earliest_cr_year": 2003,
    "emp_length_missing": 0,
    "num_tl_120dpd_2m_missing": 0,
    "issue_month_num": 48,
}


DTI_LOAN_VALUES = {
    **DEFAULT_LOAN_VALUES,
    "installment": 407.0,
    "pymnt_plan": 0,
    "earliest_cr_line_month_num": 240,
}


def load_artifact(path):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Model dependency is not installed: {exc.name}")


@pytest.fixture(scope="session")
def default_bundle():
    return load_artifact(DEFAULT_MODEL_PATH)


@pytest.fixture(scope="session")
def dti_artifacts():
    return {
        "model": load_artifact(DTI_MODEL_PATH),
        "features": load_artifact(DTI_FEATURES_PATH),
        "cat_features": load_artifact(DTI_CAT_COLS_PATH),
        "config": load_artifact(DTI_CONFIG_PATH),
    }


def make_default_scoring_frame(bundle, rows):
    features = bundle["features"]
    cat_features = bundle["cat_features"]

    records = []
    for overrides in rows:
        values = {feature: 0 for feature in features}
        values.update(DEFAULT_LOAN_VALUES)
        values.update(overrides)
        records.append(values)

    frame = pd.DataFrame(records, columns=features)
    for col in cat_features:
        frame[col] = frame[col].astype("category")

    return frame


def make_dti_scoring_frame(artifacts, overrides=None):
    features = artifacts["features"]
    cat_features = artifacts["cat_features"]

    values = {feature: 0 for feature in features}
    values.update(DTI_LOAN_VALUES)
    if overrides:
        values.update(overrides)

    frame = pd.DataFrame([values], columns=features)
    for col in cat_features:
        frame[col] = frame[col].astype("category")

    return frame


def test_expected_model_artifacts_exist():
    expected_paths = [
        DEFAULT_MODEL_PATH,
        MODELS_DIR / "catboost_model.pkl",
        MODELS_DIR / "lightgbm_model.pkl",
        MODELS_DIR / "lightgbm_iso_model.pkl",
        DTI_MODEL_PATH,
        DTI_FEATURES_PATH,
        DTI_CAT_COLS_PATH,
        DTI_CONFIG_PATH,
    ]

    for path in expected_paths:
        assert path.exists(), f"Missing model artifact: {path.name}"
        assert path.stat().st_size > 0, f"Model artifact is empty: {path.name}"


def test_default_model_bundle_has_deployment_contract(default_bundle):
    assert {"model", "features", "cat_features", "threshold"} <= set(default_bundle)

    model = default_bundle["model"]
    features = default_bundle["features"]
    cat_features = default_bundle["cat_features"]
    threshold = default_bundle["threshold"]

    assert hasattr(model, "predict_proba")
    assert features
    assert len(features) == len(set(features))
    assert set(cat_features).issubset(features)
    assert 0 < threshold < 1
    assert threshold == pytest.approx(0.2625)

    if hasattr(model, "n_features_in_"):
        assert model.n_features_in_ == len(features)


def test_default_model_features_exclude_post_origination_leakage(default_bundle):
    features = set(default_bundle["features"])

    assert features.isdisjoint(LEAKAGE_COLUMNS)


def test_default_model_scores_single_loan_with_valid_probability(default_bundle):
    frame = make_default_scoring_frame(default_bundle, rows=[{}])

    assert list(frame.columns) == default_bundle["features"]
    assert all(str(frame[col].dtype) == "category" for col in default_bundle["cat_features"])

    probabilities = default_bundle["model"].predict_proba(frame)[:, 1]
    predictions = (probabilities >= default_bundle["threshold"]).astype(int)

    assert probabilities.shape == (1,)
    assert np.isfinite(probabilities).all()
    assert np.all((0 <= probabilities) & (probabilities <= 1))
    assert set(predictions).issubset({0, 1})


def test_default_model_ranks_high_risk_profile_above_low_risk_profile(default_bundle):
    low_risk = {
        "loan_amnt": 8000,
        "term": 36,
        "int_rate": 7.5,
        "grade": "A",
        "sub_grade": "A2",
        "annual_inc": 120000,
        "dti": 8.0,
        "fico_range_low": 760,
        "revol_util": 15.0,
        "bc_util": 10.0,
        "percent_bc_gt_75": 0.0,
        "acc_open_past_24mths": 1,
    }
    high_risk = {
        "loan_amnt": 28000,
        "term": 60,
        "int_rate": 25.0,
        "grade": "F",
        "sub_grade": "F4",
        "annual_inc": 38000,
        "dti": 38.0,
        "fico_range_low": 660,
        "revol_util": 92.0,
        "bc_util": 95.0,
        "percent_bc_gt_75": 90.0,
        "delinq_2yrs": 3,
        "pub_rec": 2,
        "acc_open_past_24mths": 10,
        "inq_last_6mths": 4,
        "inq_last_12m": 8,
        "num_accts_ever_120_pd": 2,
        "mths_since_recent_revol_delinq": 2,
    }
    frame = make_default_scoring_frame(default_bundle, rows=[low_risk, high_risk])

    low_probability, high_probability = default_bundle["model"].predict_proba(frame)[:, 1]

    assert high_probability > low_probability
    assert low_probability < default_bundle["threshold"]
    assert high_probability >= default_bundle["threshold"]


def test_dti_imputation_artifacts_are_feature_aligned(dti_artifacts):
    model = dti_artifacts["model"]
    features = dti_artifacts["features"]
    cat_features = dti_artifacts["cat_features"]
    config = dti_artifacts["config"]

    assert features
    assert len(features) == len(set(features))
    assert set(cat_features).issubset(features)
    assert config["dti_lower_bound"] == 0
    assert config["dti_upper_bound"] == 100
    assert set(config["drop_cols"]).isdisjoint(features)

    if hasattr(model, "n_features_in_"):
        assert model.n_features_in_ == len(features)


def test_dti_regressor_predicts_value_inside_cleaning_bounds(dti_artifacts):
    frame = make_dti_scoring_frame(dti_artifacts)
    prediction = dti_artifacts["model"].predict(frame)

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    assert dti_artifacts["config"]["dti_lower_bound"] <= prediction[0]
    assert prediction[0] <= dti_artifacts["config"]["dti_upper_bound"]


def test_missing_data_helpers_drop_columns_and_rows_by_threshold():
    column_df = pd.DataFrame(
        {
            "mostly_missing": [np.nan, np.nan, 1.0],
            "half_missing": [1.0, np.nan, 3.0],
            "complete": [1.0, 2.0, 3.0],
        }
    )

    cleaned_columns, dropped = drop_cols_by_missing(
        column_df,
        threshold=0.5,
        return_dropped=True,
        verbose=False,
    )

    assert dropped == ["mostly_missing"]
    assert list(cleaned_columns.columns) == ["half_missing", "complete"]

    row_df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.nan],
            "b": [1.0, 2.0, np.nan],
            "c": [1.0, 3.0, np.nan],
        }
    )

    cleaned_rows = drop_rows_by_missing(row_df, max_missing_frac=0.5, verbose=False)

    assert cleaned_rows.index.tolist() == [0, 1]

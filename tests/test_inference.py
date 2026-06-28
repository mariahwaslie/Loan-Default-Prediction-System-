import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

joblib = pytest.importorskip("joblib")


ROOT = Path(__file__).resolve().parents[1]
MODEL_BUNDLE_PATH = ROOT / "models" / "loan_default_model.pkl"


BASE_LOAN = {
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


@pytest.fixture(scope="session")
def model_bundle():
    if not MODEL_BUNDLE_PATH.exists():
        pytest.skip("Default model bundle has not been generated yet.")
    try:
        return joblib.load(MODEL_BUNDLE_PATH)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Model dependency is not installed: {exc.name}")


def make_scoring_frame(bundle, rows):
    records = []
    for overrides in rows:
        record = {feature: 0 for feature in bundle["features"]}
        record.update(BASE_LOAN)
        record.update(overrides)
        records.append(record)

    frame = pd.DataFrame(records, columns=bundle["features"])

    for column in bundle["cat_features"]:
        frame[column] = frame[column].astype("category")

    return frame


def test_model_bundle_supports_probability_inference(model_bundle):
    frame = make_scoring_frame(model_bundle, rows=[{}])

    probabilities = model_bundle["model"].predict_proba(frame)

    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_inference_uses_saved_threshold_for_binary_prediction(model_bundle):
    frame = make_scoring_frame(model_bundle, rows=[{}])

    default_probability = model_bundle["model"].predict_proba(frame)[:, 1]
    prediction = (default_probability >= model_bundle["threshold"]).astype(int)

    assert model_bundle["threshold"] == pytest.approx(0.2625)
    assert prediction.dtype.kind in {"i", "u"}
    assert set(prediction).issubset({0, 1})


def test_inference_frame_preserves_feature_order_and_categorical_dtypes(model_bundle):
    frame = make_scoring_frame(model_bundle, rows=[{}])

    assert list(frame.columns) == model_bundle["features"]
    assert all(str(frame[column].dtype) == "category" for column in model_bundle["cat_features"])


def test_inference_fails_fast_when_required_features_are_missing(model_bundle):
    frame = make_scoring_frame(model_bundle, rows=[{}])
    incomplete_frame = frame.drop(columns=[model_bundle["features"][0]])

    with pytest.raises(Exception):
        model_bundle["model"].predict_proba(incomplete_frame)


def test_model_scores_directionally_riskier_profile_higher(model_bundle):
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
    frame = make_scoring_frame(model_bundle, rows=[low_risk, high_risk])

    low_probability, high_probability = model_bundle["model"].predict_proba(frame)[:, 1]

    assert high_probability > low_probability